import json
import re

INPUT_FILE = "privacybench_results-qwen2.53b.json"
OUTPUT_FILE = "privacybench_numeric_scores-qwen2.53b.json"

def convert_grade_to_percentage(grade_str):
    """
    Extracts the numeric grade from a string like "Grade: 4\n\nJustification: ..."
    and converts it to a percentage score (1-100) based on a 0-to-5 scale.
    """
    match = re.search(r"Grade:\s*([0-9]+(?:\.[0-9]+)?)", grade_str)
    if match:
        grade = float(match.group(1))
        # Convert grade from 0-5 to a percentage (0-100).
        percentage = (grade / 5.0) * 100
        return percentage
    return None

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
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

    # Optionally, compute an overall average score across entries.
    if overall_scores:
        overall_average = sum(overall_scores) / len(overall_scores)
        print(f"Overall Average Score: {overall_average:.2f}")
    else:
        print("No valid numeric scores found.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Updated results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
