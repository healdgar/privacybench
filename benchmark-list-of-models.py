import os
import json
import re
import glob
import time
import requests
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import numpy as np
from scipy.interpolate import make_interp_spline

load_dotenv(override=True)  # Force .env values to override any existing variables

# ---------------------------
# Helper Functions
# ---------------------------
def sanitize_model_name(model_name):
    """
    Sanitizes a model name for use in filenames by replacing characters
    that may not be allowed (e.g. "/" becomes "_" and ":" becomes "-").
    """
    sanitized = model_name.replace("/", "_").replace(":", "-")
    return sanitized

def extract_model_name(filename):
    # Look for pattern: score_report-<modelname>.json
    m = re.search(r"score_report-([^\.]+)\.json", filename)
    if m:
        return m.group(1)
    return "Unknown_Model"

def load_score_report(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and len(data) > 0:
        model_key = list(data.keys())[0]
        return model_key, data[model_key]
    return None, None

# ---------------------------
# Configuration and Constants
# ---------------------------
USE_ONLINE_API = False  # Set to True to test online APIs (e.g., gpt-4o)

QUESTION_SOURCE_FILE = "evaluation_questions/privacybench_PII_redaction.json"
QUESTION_SOURCE_BASENAME = os.path.splitext(os.path.basename(QUESTION_SOURCE_FILE))[0]

# Define a list of tested models (you can add more here)
TESTED_MODELS = [
    "mistral:latest",
    "nemotron-mini:latest",
    "gemma2:9b",
    "gemma2:2b",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "llama3.2:1b",
    "qwen2.5:latest",
    "llama3.2:3b",
    "llama3.2-vision:latest",
    "deepseek-r1:7b",
    "deepseek-r1:1.5b",
    "deepseek-r1:8b",
    "llama3.2:latest"
    # "another/model:version",  # Add more models as needed.
]

# Judging model configuration remains constant.
JUDGING_MODEL = "gpt-4o"
SANITIZED_JUDGING = sanitize_model_name(JUDGING_MODEL)

# Local model (for benchmarking) will be set per tested model.
# Also, the API URL for the local model (Ollama) is defined here.
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# ---------------------------
# Setup Judging Client(s)
# ---------------------------
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048,
}
gemini_model = genai.GenerativeModel(
    model_name=JUDGING_MODEL,
    generation_config=generation_config,
)

# Global filename variables (they will be updated per tested model).
LOG_FILE = ""
GRADED_FILE = ""
NUMERIC_FILE = ""
SCORE_REPORT_FILE = ""
GRADING_SUMMARY_FILE = ""

# ---------------------------
# Pipeline Functions (using global filenames)
# ---------------------------
def load_questions():
    with open(QUESTION_SOURCE_FILE, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data.get("questions", [])

def query_model(prompt):
    """
    Queries the model under test.
    If USE_ONLINE_API is True, uses the OpenAI API; otherwise, uses the local Ollama API.
    Relies on the global variable LOCAL_MODEL_NAME.
    """
    if USE_ONLINE_API:
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=messages,
                response_format={"type": "text"},
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        payload = {"model": LOCAL_MODEL_NAME, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Error: {response.status_code}"

def run_benchmark():
    print("Running benchmark on model under test...")
    questions = load_questions()
    results = []
    for question in questions:
        start_time = time.time()
        prompt = f"Question: {question['question']}\nAnswer:"
        response_text = query_model(prompt)
        runtime = round(time.time() - start_time, 2)
        results.append({
            "id": question["id"],
            "question": question["question"],
            "model_response": response_text,
            "model_used": LOCAL_MODEL_NAME,
            "runtime_seconds": runtime,
            "correct_answer": question.get("correct_answer", "N/A"),
            "type": question["type"]
        })
        print(f"Processed Question {question['id']} - Runtime: {runtime}s")
        time.sleep(1)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Benchmark results saved to {LOG_FILE}")

def query_grading_model(question, model_response, correct_answer, question_type):
    grading_prompt = f"""
You are grading an AI's response for accuracy.

Question Type: {question_type}
Question: {question}
Expected Correct Answer: {correct_answer}
Model's Response: {model_response}

Grade the response from 0 to 5:
- 5 = Perfectly correct.
- 4 = Mostly correct, minor mistake.
- 3 = Partially correct, missing key details.
- 2 = Somewhat relevant, but mostly incorrect.
- 1 = Poor response, incorrect.
- 0 = Not related to the question.

Provide a 1-sentence justification.
    """
    if JUDGING_MODEL.startswith("gpt-"):
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": grading_prompt}
        ]
        try:
            response = client.chat.completions.create(
                model=JUDGING_MODEL,
                messages=messages,
                response_format={"type": "text"},
                temperature=0.2,
                max_completion_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            print(f"[DEBUG] Raw grading response for question '{question[:50]}...':")
            print(response)
            raw_text = response.choices[0].message.content
            print(f"[DEBUG] Raw text extracted: {raw_text}")
            return raw_text.strip()
        except Exception as e:
            print("Error using OpenAI GPT judging model:", e)
            return f"Grading Error: {str(e)}"
    else:
        try:
            response = gemini_model.generate_content(grading_prompt)
            print(f"[DEBUG] Raw Gemini response for question '{question[:50]}...':")
            print(response)
            return response.text.strip()
        except Exception as e:
            print("Error using Gemini judging model:", e)
            return f"Grading Error: {str(e)}"

def run_grading():
    print("Running grading using judging model...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)
    graded_results = []
    total = len(responses)
    for idx, entry in enumerate(responses, start=1):
        print(f"[{datetime.now()}] Processing question {entry['id']} ({idx}/{total})...")
        start_time = time.time()
        grade = query_grading_model(
            entry["question"],
            entry["model_response"],
            entry["correct_answer"],
            entry["type"]
        )
        elapsed = time.time() - start_time
        print(f"[{datetime.now()}] Completed question {entry['id']} in {elapsed:.2f}s")
        graded_results.append({
            "id": entry["id"],
            "question": entry["question"],
            "model_response": entry["model_response"],
            "score": grade,
            "model_used": entry["model_used"],
            "runtime_seconds": entry["runtime_seconds"]
        })
    with open(GRADED_FILE, "w", encoding="utf-8") as f:
        json.dump(graded_results, f, indent=4)
    print(f"\n✅ Graded results saved to {GRADED_FILE}")

def convert_grade_to_percentage(grade_str):
    match = re.search(r"Grade:\s*([0-9]+(?:\.[0-9]+)?)", grade_str)
    if not match:
        match = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)", grade_str)
    if match:
        grade = float(match.group(1))
        return (grade / 5.0) * 100
    return None

def convert_grades_to_numeric():
    print("Converting grades to numeric scores...")
    with open(GRADED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    overall_scores = []
    for entry in data:
        original_score_text = entry.get("score", "")
        numeric_score = convert_grade_to_percentage(original_score_text)
        if numeric_score is not None:
            entry["numeric_score"] = numeric_score
            overall_scores.append(numeric_score)
        else:
            entry["numeric_score"] = None
    if overall_scores:
        overall_average = sum(overall_scores) / len(overall_scores)
        print(f"Overall Average Score: {overall_average:.2f}")
    else:
        print("No valid numeric scores found.")
    with open(NUMERIC_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Numeric scores saved to {NUMERIC_FILE}")

def extract_numeric_score(entry):
    if "numeric_score" in entry and entry["numeric_score"] is not None:
        return entry["numeric_score"]
    score_text = entry.get("score", "")
    match = re.search(r"Grade:\s*([0-9]+(?:\.[0-9]+)?)", score_text)
    if match:
        grade = float(match.group(1))
        return (grade / 5.0) * 100
    return None

def load_entries_from_files(prefix):
    entries = []
    for filename in glob.glob(f"{prefix}*.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    entries.extend(data)
                else:
                    print(f"Warning: {filename} does not contain a JSON list.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return entries

def compile_score_report():
    print("Compiling final score report...")
    prefix = f"privacybench_results-{SANITIZED_TESTED}-{SANITIZED_JUDGING}-{QUESTION_SOURCE_BASENAME}"
    results_entries = load_entries_from_files(prefix)
    numeric_entries = load_entries_from_files(f"privacybench_numeric_scores-{SANITIZED_TESTED}-{SANITIZED_JUDGING}-{QUESTION_SOURCE_BASENAME}")
    all_entries = results_entries + numeric_entries
    if not all_entries:
        print("No entries found for score report.")
        return

    model_data = defaultdict(lambda: defaultdict(lambda: {"scores": [], "question": None}))
    for entry in all_entries:
        numeric_score = extract_numeric_score(entry)
        if numeric_score is None:
            continue
        question_id = entry.get("id")
        model_used = entry.get("model_used", "Unknown Model")
        question_text = entry.get("question", "No question text provided.")
        if question_id is None:
            continue
        model_entry = model_data[model_used][question_id]
        model_entry["scores"].append(numeric_score)
        if model_entry["question"] is None:
            model_entry["question"] = question_text

    report = {}
    for model, questions in model_data.items():
        breakdown = {}
        all_scores = []
        for q in range(1, 26):
            if q in questions and questions[q]["scores"]:
                avg_score = sum(questions[q]["scores"]) / len(questions[q]["scores"])
                breakdown[q] = {
                    "score": round(avg_score, 2),
                    "question": questions[q]["question"]
                }
                all_scores.append(avg_score)
            else:
                breakdown[q] = {"score": None, "question": None}
        overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else None
        report[model] = {
            "overall_score": overall,
            "question_breakdown": breakdown
        }
    with open(SCORE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"✅ Score report saved to {SCORE_REPORT_FILE}")

def run_grading_summary():
    print("Running grading summary...")
    try:
        with open(GRADED_FILE, "r", encoding="utf-8") as f:
            graded_results = json.load(f)
    except Exception as e:
        print("Error loading graded results:", e)
        return

    graded_results.sort(key=lambda x: x.get("id", 0))
    
    summary_lines = []
    summary_lines.append(f"Graded Results for Tested Model '{TESTED_MODEL}':\n")
    for entry in graded_results:
        qid = entry.get("id", "N/A")
        question_text = entry.get("question", "No question text provided.")
        score_text = entry.get("score", "No score provided.")
        summary_lines.append(f"Q{qid}: {question_text}\nScore & Justification: {score_text}\n")
    summary_lines.append("\nBased on the above 25 graded results, please provide a brief narrative about the model's overall performance, commenting on any trends, tendencies, or subject matter blind spots that the results may indicate.")
    
    summary_prompt = "\n".join(summary_lines)
    
    if JUDGING_MODEL.startswith("gpt-"):
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": summary_prompt}
        ]
        try:
            response = client.chat.completions.create(
                model=JUDGING_MODEL,
                messages=messages,
                response_format={"type": "text"},
                temperature=0.2,
                max_completion_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            narrative = response.choices[0].message.content.strip()
        except Exception as e:
            print("Error using OpenAI GPT for grading summary:", e)
            narrative = f"Grading Summary Error: {str(e)}"
    else:
        try:
            response = gemini_model.generate_content(summary_prompt)
            narrative = response.text.strip()
        except Exception as e:
            print("Error using Gemini for grading summary:", e)
            narrative = f"Grading Summary Error: {str(e)}"
    
    try:
        with open(GRADING_SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(narrative)
        print(f"\n✅ Grading summary saved to {GRADING_SUMMARY_FILE}")
    except Exception as e:
        print("Error saving grading summary:", e)

# ---------------------------
# Pipeline Function for a Tested Model
# ---------------------------
def run_pipeline_for_model(tested_model):
    global LOG_FILE, GRADED_FILE, NUMERIC_FILE, SCORE_REPORT_FILE, GRADING_SUMMARY_FILE, LOCAL_MODEL_NAME, TESTED_MODEL, SANITIZED_TESTED
    TESTED_MODEL = tested_model  # update tested model
    SANITIZED_TESTED = sanitize_model_name(tested_model)
    LOG_FILE = f"privacybench_log-{SANITIZED_TESTED}-{QUESTION_SOURCE_BASENAME}.json"
    GRADED_FILE = f"privacybench_results-{SANITIZED_TESTED}-{SANITIZED_JUDGING}-{QUESTION_SOURCE_BASENAME}.json"
    NUMERIC_FILE = f"privacybench_numeric_scores-{SANITIZED_TESTED}-{SANITIZED_JUDGING}-{QUESTION_SOURCE_BASENAME}.json"
    SCORE_REPORT_FILE = f"privacybench_score_report-{SANITIZED_TESTED}-{SANITIZED_JUDGING}-{QUESTION_SOURCE_BASENAME}.json"
    GRADING_SUMMARY_FILE = f"grading_summary-{SANITIZED_TESTED}-{SANITIZED_JUDGING}-{QUESTION_SOURCE_BASENAME}.txt"
    
    # Update the global local model name.
    LOCAL_MODEL_NAME = tested_model
    
    print(f"=== Running pipeline for model: {tested_model} ===")
    run_benchmark()
    run_grading()
    convert_grades_to_numeric()
    compile_score_report()
    run_grading_summary()
    print(f"=== Pipeline completed for model: {tested_model} ===\n")


# ---------------------------
# Main Execution Flow
# ---------------------------
if __name__ == "__main__":
    start_time = datetime.now()
    print("=== PrivacyBench Evaluation Pipeline Started ===")
    
    # Iterate over the list of tested models.
    for model in TESTED_MODELS:
        run_pipeline_for_model(model)
    
    end_time = datetime.now()
    print("=== Pipeline Completed in:", end_time - start_time, "===")
