import os
import json
import glob
from collections import defaultdict

# Define file patterns and corresponding keys to check.
# - For "logs": check "model_response"
# - For "numeric scores": check "numeric_score"
# - For "results": check "score"
FILE_PATTERNS = {
    "log": "privacybench_log-*privacybench_PII_filtering*.json",
    "numeric": "privacybench_numeric_scores-*privacybench_PII_filtering*.json",
    "results": "privacybench_results-*privacybench_PII_filtering*.json"
}


KEY_MAP = {
    "log": "model_response",
    "numeric": "numeric_score",
    "results": "score"
}

# Dictionaries to hold error counts, total entries, and filenames per question id.
error_counts = defaultdict(lambda: defaultdict(int))  # {file_type: {question_id: error_count}}
total_counts = defaultdict(lambda: defaultdict(int))  # {file_type: {question_id: total_entries}}
error_files = defaultdict(lambda: defaultdict(set))     # {file_type: {question_id: set(filenames)}}

def is_null_value(value):
    """Return True if the value is None or a string that equals 'null' (case-insensitive)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() == "null":
        return True
    return False

# Iterate through each file pattern.
for file_type, pattern in FILE_PATTERNS.items():
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Assume each file is a list of log entries.
            for entry in data:
                qid = entry.get("id", "Unknown")
                total_counts[file_type][qid] += 1
                key_to_check = KEY_MAP[file_type]
                value = entry.get(key_to_check)
                if is_null_value(value):
                    error_counts[file_type][qid] += 1
                    error_files[file_type][qid].add(os.path.basename(filepath))
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

# Print summary of errors per file type.
print("Summary of Null Values in Log Files:")
for file_type in FILE_PATTERNS.keys():
    print(f"\nFile Type: {file_type.upper()}")
    for qid, errors in sorted(error_counts[file_type].items(), key=lambda x: x[0]):
        total = total_counts[file_type].get(qid, 0)
        pct = (errors / total * 100) if total > 0 else 0
        filenames = sorted(list(error_files[file_type][qid]))
        print(f"  Question ID {qid}: {errors} errors out of {total} entries ({pct:.2f}%)")
        print(f"    Files: {', '.join(filenames)}")
