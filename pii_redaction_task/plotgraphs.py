import os
import re
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------------------------
# Helper Functions
# ---------------------------

# Function to extract the model name from filename.
# Assumes filename pattern like "score_report-<modelname>.json"
def extract_model_name(filename):
    m = re.search(r"score_report-([^\.]+)\.json", filename)
    if m:
        return m.group(1)
    return "Unknown_Model"

# Function to load overall score and question breakdown from a JSON file.
def load_score_report(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and len(data) > 0:
        model_key = list(data.keys())[0]
        return model_key, data[model_key]
    return None, None

# Gather all score report files that match the "privacybench_PII_filtering" question set.
files = glob.glob("*score_report-*privacybench_PII_redaction*.json")
if not files:
    print("No score report files found.")
    exit(1)

# Dictionaries to hold overall scores and per-question scores per model.
overall_scores = {}      # { model_name: overall_score }
per_question_scores = {} # { model_name: [score1, score2, ..., score25] }

# Process each file
for filepath in files:
    model_from_filename = extract_model_name(os.path.basename(filepath))
    model_key, report = load_score_report(filepath)
    if report is None:
        continue
    model_name = model_key if model_key != "Unknown_Model" else model_from_filename
    overall_score = report.get("overall_score", None)
    overall_scores[model_name] = overall_score

    breakdown = report.get("question_breakdown", {})
    q_scores = []
    for q in range(1, 26):
        score_entry = breakdown.get(str(q)) or breakdown.get(q)
        if score_entry and score_entry.get("score") is not None:
            q_scores.append(score_entry["score"])
        else:
            q_scores.append(0)
    per_question_scores[model_name] = q_scores

# ---- Plot Overall Scores Bar Graph (Ordered by Score) ----
plt.figure(figsize=(10, 6))
sorted_models = sorted(overall_scores, key=lambda m: overall_scores[m], reverse=True)
sorted_scores = [overall_scores[m] for m in sorted_models]
colors = ['red' if 'gpt-4o' in m.lower() else 'skyblue' for m in sorted_models]

bars = plt.bar(sorted_models, sorted_scores, color=colors)
plt.xlabel("Model")
plt.ylabel("Overall Score (0-100)")
plt.title("Model Accuracy in Identifying and Redacting Personal Data")
plt.ylim(0, 100)
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("overall_scores_bar_graph.png")
plt.show()

# ---- Plot Average Score per Question Bar Graph (Difficulty Metric) ----
# This graph aggregates the per-question scores across all models.
plt.figure(figsize=(10, 6))
question_avgs = []
for q in range(1, 26):
    scores = []
    for model in per_question_scores:
        # q-1 because our list index starts at 0.
        scores.append(per_question_scores[model][q-1])
    avg_score = sum(scores) / len(scores) if scores else 0
    question_avgs.append(avg_score)

x = list(range(1, 26))
plt.bar(x, question_avgs, color='purple')
plt.xlabel("Question Number")
plt.ylabel("Average Score (0-100)")
plt.title("Average Score per Question (Difficulty Metric)")
plt.ylim(0, 100)
for i, avg in enumerate(question_avgs):
    plt.text(i+1, avg + 2, f"{avg:.1f}", ha='center', va='bottom', fontsize=9)
plt.xticks(x)
plt.tight_layout()
plt.savefig("average_score_per_question.png")
plt.show()

# ---- Plot Per-Question Performance Curve Graph ----
plt.figure(figsize=(12, 6))
questions = np.array(list(range(1, 26)))
for model, q_scores in per_question_scores.items():
    y = np.array(q_scores)
    x_smooth = np.linspace(questions.min(), questions.max(), 300)
    try:
        spline = make_interp_spline(questions, y, k=2)
        y_smooth = spline(x_smooth)
    except Exception as e:
        print(f"Could not interpolate for model {model}: {e}")
        x_smooth = questions
        y_smooth = y
    color = 'red' if 'gpt-4o' in model.lower() else None
    plt.plot(x_smooth, y_smooth, label=model, color=color)
plt.xlabel("Question Number")
plt.ylabel("Score (0-100)")
plt.title("Per-Question Performance of Models")
plt.xticks(list(range(1, 26)))
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig("per_question_performance.png")
plt.show()
