# PrivacyBench

Repository dedicated to benchmarking privacy-related performance of generative AI models.

**Problem to be Solved**

- Software and analytics businesses seek to run machine learning and business analytics on usage data in order to improve their products and services, but usage data almost always contains personal data.
- This exposes businesses to potential regulatory fines, contractual violations, reputational harm, and increased data breach risk, and exposes individuals to unnecessary risk.
- No method of personal data filtering is likely to be "perfect", thus a method of benchmarking filtering and redaction by LLM technology is needed.

**Proposed Solution**

- Develop a testing method for benchmarking performance in personal data redaction.
- Develop and report methods on LLM performance.
- Identify, in particular, LLM models that can be deployed locally and efficiently for maximum privacy and feasibility.

**Context of Problem (both legal and technical)**

- Privacy laws have broad definitions of personal data that indicate that any information that relates to an identified or identifiable living human is personal data.
- Traditional regex-based pattern searching is insufficiently flexible to identify personal data, as whether a person can be identified is related to the context: an area where LLMs excel.
- Privacy Enhancing Technology, particularly using AI, should be effective, useful, purpose-built, and trustworthy.

PrivacyBench Results

![Average score per question]\(overall_scores_bar_graph.png)

**Explanation**

In this figure, gpt-4o is used as the overall judge of the output of both itself and the other models.  Gpt-4o, being a much larger and more intelligent model than locally-hosted models of 8B parameters and smaller, is a reasonable judge of performance.

**Observations**

- Gemma2:9B shows remarkable performance given its size.  It nearly matches GPT-4o's performance at this task, despite being a much smaller model that can be run on the modest processing resources of a decent workstation.  This makes it a possible good candidate for additional fine-tuning to the specific task to further improve its scoring.  It's most significant error is that identified a generic email address as person data ([info@example.com](mailto\:info@example.com)) and redacted it, which is a conservative error.
- Gpt-4o performed very well, but interestingly found flaws in its own output.
- To review AI-generated summaries of each model's performance see the pii\_redaction\_task folder in the repo.

## Benchmarking Methodology - Personal Data Redaction

1. **Question Set Preparation**

   - We maintain a JSON file (`privacybench_PII_redaction.json`) containing 25 test questions. Each question prompts the model to perform redaction of personal data.
   - Each question provides a reference `correct_answer` for grading purposes, so we can compare how well the model’s redactions matched the expected approach.

2. **Execution of Tested Models**

   - The script defines a list of `TESTED_MODELS`, each identified by a model name or endpoint.
   - For every model in that list, a pipeline function (`run_pipeline_for_model()`) runs each question in `privacybench_PII_redaction.json` against the model.
   - The model is prompted with a fixed format that includes instructions ("Examine the text excerpt and redact...").
   - The script logs each model’s response, along with the runtime, to a JSON file.

3. **Judging Model (Grading)**

   - A separate model, labeled `JUDGING_MODEL` (e.g. `gpt-4o`), is used to score each response.
   - The script passes the question, the model’s response, and the expected answer to the judging model with a standardized grading prompt.
   - The judging model outputs a rating from 0 to 5 and provides a short justification. That raw text is captured and stored.

4. **Conversion to Numeric Scores**

   - A function parses the judging model’s textual score, extracting the numeric grade (0–5).
   - The script converts that 0–5 scale into a 0–100 range by `(grade / 5.0) * 100`.
   - The numeric results are stored in a separate JSON file, making it easier to visualize or compare performance.

5. **Aggregation and Reporting**

   - The script compiles all tested model results into an overall "score report," which captures both per-question results and an overall average for each model.
   - The final script (`compile_score_report()`) creates a JSON file containing each model’s average score, as well as a breakdown of each of the 25 questions.
   - Optionally, `run_grading_summary()` uses the same judging model to produce a short narrative about each model’s strengths and weaknesses, based on the aggregated results.

6. **Result Visualization**

   - A separate script handles plotting the results (`matplotlib`-based bar charts and line graphs). It reads the "score\_report-\*.json" files and generates:
     - **Overall Scores Bar Graph**: showing each model’s total average score.
     - **Average Score per Question**: illustrating question difficulty or how well most models handled a question.
     - **Per-Question Performance Curves**: a spline-interpolated line chart for each model, highlighting performance patterns across the 25 questions.

**What It Demonstrates**

- How to orchestrate a repeatable pipeline that tests multiple AI models.
- How to use a specialized "judge" model to systematically evaluate and score each model’s performance.
- How to automate the entire workflow, from generating answers to compiling final performance summaries, ensuring consistent measurement across models.

**Caveats**

- No aspect of this repository constitutes legal advice.
- The performance benchmark is influenced by the correctness of the grading model.
- Real-world personal data redaction remains context-dependent, and no automated approach is guaranteed to be 100% correct.

**Summary**

- This workflow provides a systematic approach to evaluating how well AI models redact personal data.
- By standardizing questions, collecting responses, using a single judging model, and generating visual reports, we can compare relative performance across multiple large language models.


