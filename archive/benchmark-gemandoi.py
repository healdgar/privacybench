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

load_dotenv(override=True)
  # load API keys from .env file

# ---------------------------
# Configuration and Constants
# ---------------------------
# Question source file and its base name (without extension)
QUESTION_SOURCE_FILE = "privacybench_questions.json"
QUESTION_SOURCE_BASENAME = os.path.splitext(os.path.basename(QUESTION_SOURCE_FILE))[0]

# Tested model (the local model used for benchmarking)
TESTED_MODEL = "qwen2.5-3b"  

# Local model (Ollama) configuration for benchmark
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL_NAME = "qwen2.5:3b"  # Used for querying the local model

# Judging model configuration.
# To switch between a Gemini model and an OpenAI GPT model (e.g., "gpt-4o"),
# simply change the value of JUDGING_MODEL.
JUDGING_MODEL = "gpt-4o"  
# For Gemini, you might use: JUDGING_MODEL = "gemini-2.0-flash-thinking-exp-01-21"

# ---------------------------
# Setup Judging Client(s)
# ---------------------------
# For OpenAI GPT-based judging model:
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()

# For Gemini-based judging model:
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

# ---------------------------
# Define output filename templates
# ---------------------------
LOG_FILE = f"privacybench_log-{TESTED_MODEL}-{QUESTION_SOURCE_BASENAME}.json"
GRADED_FILE = f"privacybench_results-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}.json"
NUMERIC_FILE = f"privacybench_numeric_scores-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}.json"
SCORE_REPORT_FILE = f"privacybench_score_report-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}.json"

# ---------------------------
# Step 1: Run Benchmark (Local Model)
# ---------------------------
def load_questions():
    with open(QUESTION_SOURCE_FILE, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data.get("questions", [])

def query_local_model(prompt):
    payload = {"model": LOCAL_MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return f"Error: {response.status_code}"

def run_benchmark():
    print("Running benchmark on local model...")
    questions = load_questions()
    results = []
    for question in questions:
        start_time = time.time()
        prompt = f"Question: {question['question']}\nAnswer:"
        response_text = query_local_model(prompt)
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

# ---------------------------
# Step 2: Run Grading (Judging Model)
# ---------------------------
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
    # Use OpenAI's client if the judging model starts with "gpt-"
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
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error using OpenAI GPT judging model:", e)
            return f"Grading Error: {str(e)}"
    else:
        # Otherwise, assume Gemini is being used.
        try:
            response = gemini_model.generate_content(grading_prompt)
            return response.text.strip()
        except Exception as e:
            print("Error using Gemini judging model:", e)
            return f"Grading Error: {str(e)}"

def run_grading():
    print("Running grading using judging model...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)
    graded_results = []
    for entry in responses:
        grade = query_grading_model(
            entry["question"],
            entry["model_response"],
            entry["correct_answer"],
            entry["type"]
        )
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

# ---------------------------
# Step 3: Convert Grades to Numeric Scores
# ---------------------------
def convert_grade_to_percentage(grade_str):
    match = re.search(r"Grade:\s*([0-9]+(?:\.[0-9]+)?)", grade_str)
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

# ---------------------------
# Step 4: Compile Score Report
# ---------------------------
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
    prefix = f"privacybench_results-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}"
    results_entries = load_entries_from_files(prefix)
    numeric_entries = load_entries_from_files(f"privacybench_numeric_scores-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}")
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

# ---------------------------
# Main Execution Flow
# ---------------------------
if __name__ == "__main__":
    start_time = datetime.now()
    print("=== PrivacyBench Evaluation Pipeline Started ===")
    
    # Step 1: Benchmark local model responses
    #run_benchmark()
    
    # Step 2: Grade the responses using the judging model (OpenAI GPT or Gemini)
    #run_grading()
    
    # Step 3: Convert graded responses to numeric scores
    convert_grades_to_numeric()
    
    # Step 4: Compile final score report
    #compile_score_report()
    
    end_time = datetime.now()
    print("=== Pipeline Completed in:", end_time - start_time, "===")
