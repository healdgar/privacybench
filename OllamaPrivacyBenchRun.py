import json
import requests
import time
import os
from datetime import datetime

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

# Load benchmark questions
with open("privacybench_questions.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)
    QUESTIONS = data["questions"]

# Query local model
def query_ollama(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_API_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return f"Error: {response.status_code}"

# Run benchmark and save results
def run_benchmark():
    log_file = "privacybench_log_qwen2.53b.json"
    results = []

    for question in QUESTIONS:
        start_time = time.time()

        prompt = f"Question: {question['question']}\nAnswer:"
        response = query_ollama(prompt)

        end_time = time.time()
        runtime = round(end_time - start_time, 2)

        results.append({
            "id": question["id"],
            "question": question["question"],
            "model_response": response,
            "model_used": MODEL_NAME,
            "runtime_seconds": runtime,
            "correct_answer": question.get("correct_answer", "N/A"),  # Keep for grading
            "type": question["type"]  # Include question type for grading later
        })

        print(f"Processed Question {question['id']} - Runtime: {runtime}s")

        time.sleep(1)  # Avoid overwhelming the local model

    # Save log file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Results saved to {log_file} - Ready for grading.")

# Execute benchmark
if __name__ == "__main__":
    run_benchmark()
