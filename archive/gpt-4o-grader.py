import json
from openai import OpenAI

# Instantiate the client with your API key
client = OpenAI(api_key="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")

# Define the external grading model
EXTERNAL_GRADING_MODEL = "gpt-4o"  # Replace with your API model

# Load saved responses
log_file = "privacybench_log_qwen2.53b.json"
with open(log_file, "r", encoding="utf-8") as f:
    responses = json.load(f)

# Function to query the grading model
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
    # Build the messages list following the example format
    messages = [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": grading_prompt}
    ]

    completion = client.chat.completions.create(
        model=EXTERNAL_GRADING_MODEL,
        messages=messages,
        response_format={"type": "text"},
        temperature=0.2,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    try:
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Grading Error: {str(e)}"

# Process and grade responses
graded_results = []
for entry in responses:
    score = query_grading_model(
        entry["question"],
        entry["model_response"],
        entry["correct_answer"],
        entry["type"]
    )

    graded_results.append({
        "id": entry["id"],
        "question": entry["question"],
        "model_response": entry["model_response"],
        "score": score,
        "model_used": entry["model_used"],
        "runtime_seconds": entry["runtime_seconds"]
    })

# Save graded results
graded_file = "privacybench_results-qwen2.53b.json"
with open(graded_file, "w", encoding="utf-8") as f:
    json.dump(graded_results, f, indent=4)

print(f"\n✅ Graded results saved to {graded_file}")
