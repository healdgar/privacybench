import os
import json
import re
import glob
from collections import defaultdict

def extract_numeric_score(entry):
    """
    Returns the numeric score from an entry.
    Prefer the 'numeric_score' field if present;
    otherwise, extract from the 'score' string (e.g., "Grade: 4\n\nJustification: ...")
    and convert it from a 0–5 scale to a percentage (1–100 scale).
    """
    if "numeric_score" in entry and entry["numeric_score"] is not None:
        return entry["numeric_score"]
    
    score_text = entry.get("score", "")
    match = re.search(r"Grade:\s*([0-9]+(?:\.[0-9]+)?)", score_text)
    if match:
        grade = float(match.group(1))
        return (grade / 5.0) * 100
    return None

def load_entries_from_files(prefix):
    """Loads and returns all JSON entries from files in the current directory starting with the given prefix."""
    entries = []
    for filename in glob.glob(f"{prefix}*.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Expecting data to be a list of entries.
                if isinstance(data, list):
                    entries.extend(data)
                else:
                    print(f"Warning: {filename} does not contain a JSON list.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return entries

def compile_score_report(all_entries):
    """
    Compiles a report mapping each model (from 'model_used') to its overall average numeric score
    and a breakdown per question (IDs 1 to 25). For each question, the report includes the average score
    and the question text.
    """
    # Structure: model -> { question_id -> {"scores": [list of scores], "question": question text} }
    model_data = defaultdict(lambda: defaultdict(lambda: {"scores": [], "question": None}))
    
    for entry in all_entries:
        numeric_score = extract_numeric_score(entry)
        if numeric_score is None:
            continue  # Skip entries without a valid score.
        
        question_id = entry.get("id")
        model_used = entry.get("model_used", "Unknown Model")
        question_text = entry.get("question", "No question text provided.")
        if question_id is None:
            continue
        
        model_entry = model_data[model_used][question_id]
        model_entry["scores"].append(numeric_score)
        # If question text hasn't been set yet, assign it.
        if model_entry["question"] is None:
            model_entry["question"] = question_text

    # Build the final report
    report = {}
    for model, questions in model_data.items():
        breakdown = {}
        all_scores = []
        for q in range(1, 26):  # Assuming questions 1 through 25.
            if q in questions and questions[q]["scores"]:
                avg_score = sum(questions[q]["scores"]) / len(questions[q]["scores"])
                breakdown[q] = {
                    "score": round(avg_score, 2),
                    "question": questions[q]["question"]
                }
                all_scores.append(avg_score)
            else:
                breakdown[q] = {
                    "score": None,
                    "question": None
                }
        overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else None
        report[model] = {
            "overall_score": overall,
            "question_breakdown": breakdown
        }
    return report

def main():
    # Load entries from both types of files.
    results_entries = load_entries_from_files("privacybench_results")
    numeric_entries = load_entries_from_files("privacybench_numeric_scores")
    
    # Combine entries. If an entry appears in both sets, that's acceptable since we're averaging per question.
    all_entries = results_entries + numeric_entries

    if not all_entries:
        print("No entries found in the specified files.")
        return

    score_report = compile_score_report(all_entries)

    # Print report in a readable format.
    for model, scores in score_report.items():
        print(f"Model: {model}")
        print(f"  Overall Score: {scores['overall_score']}")
        print("  Question Breakdown:")
        for q in range(1, 26):
            qb = scores['question_breakdown'].get(q, {})
            score_val = qb.get("score")
            question_text = qb.get("question")
            print(f"    Q{q}: Score = {score_val} | Question: {question_text}")
        print("-" * 40)

    # Save the report to a JSON file.
    output_file = "privacybench_score_report.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(score_report, f, indent=4)
        print(f"✅ Score report saved to {output_file}")
    except Exception as e:
        print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()
# The script compiles a report of the overall average numeric scores and question-specific average scores for each model used in the PrivacyBench evaluation. It loads entries from JSON files containing the results and numeric scores, processes the data, and generates a report with the model names, overall scores, and question breakdowns. The report is saved to a JSON file named "privacybench_score_report.json." The script is designed to handle multiple entries per model and question, providing a comprehensive summary of the evaluation outcomes.