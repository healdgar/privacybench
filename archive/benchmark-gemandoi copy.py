import os
import json
import re
import glob
import time
import requests
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

# ---------------------------
# Configuration and Constants
# ---------------------------
# Question source file and its base name (without extension)
QUESTION_SOURCE_FILE = "privacybench_questions.json"
QUESTION_SOURCE_BASENAME = os.path.splitext(os.path.basename(QUESTION_SOURCE_FILE))[0]

# Tested model (the local model used for benchmarking)
# For naming purposes, ensure this string matches what you want in output filenames.
TESTED_MODEL = "llama3.2:1b"  

# Local model (Ollama) configuration for benchmark
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL_NAME = "llama3.2:1b"  # Used for querying the local model

# Grading model (Gemini) configuration
# Make sure you have a .env file with GEMINI_API_KEY defined, or set it in your environment.
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
JUDGING_MODEL = "gemini-2.0-flash-thinking-exp-01-21"  # Gemini model for grading

# Configure Gemini generation settings
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
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

# Define output filename templates
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
    # Assume questions are stored under the key "questions"
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
        time.sleep(1)  # Throttle requests
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Benchmark results saved to {LOG_FILE}")

# ---------------------------
# Step 2: Run Grading (Gemini Model)
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
    try:
        response = gemini_model.generate_content(grading_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Grading Error: {str(e)}"

def run_grading():
    print("Running grading using Gemini model...")
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
    # Load entries from graded results and numeric scores files
    results_entries = load_entries_from_files(f"privacybench_results-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}")
    numeric_entries = load_entries_from_files(f"privacybench_numeric_scores-{TESTED_MODEL}-{JUDGING_MODEL}-{QUESTION_SOURCE_BASENAME}")
    all_entries = results_entries + numeric_entries
    if not all_entries:
        print("No entries found for score report.")
        return

    # Structure: model -> { question_id -> {"scores": [list of scores], "question": question text} }
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
    run_benchmark()
    
    # Step 2: Grade the responses using the Gemini model
    run_grading()
    
    # Step 3: Convert graded responses to numeric scores
    convert_grades_to_numeric()
    
    # Step 4: Compile final score report
    #compile_score_report()
    
    end_time = datetime.now()
    print("=== Pipeline Completed in:", end_time - start_time, "===")
