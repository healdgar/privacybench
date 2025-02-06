import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Configure Gemini API with your API key from environment variables
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define the external grading model - using a Gemini model
EXTERNAL_GRADING_MODEL = "gemini-2.0-flash-thinking-exp-01-21"  # Or "gemini-pro-vision" if you need vision capabilities

# Configure generation settings (optional, you can adjust these)
generation_config = {
    "temperature": 0.2, # Lower temperature for more deterministic grading
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048, # Adjusted to match original openai code
}


# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name=EXTERNAL_GRADING_MODEL,
    generation_config=generation_config,
)

# Load saved responses
log_file = "privacybench_log_qwen2.53b.json"
with open(log_file, "r", encoding="utf-8") as f:
    responses = json.load(f)

# Function to query the grading model using Gemini API
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
        response = model.generate_content(grading_prompt) # Send prompt to Gemini
        return response.text.strip() # Extract text response from Gemini
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