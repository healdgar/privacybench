# privacybench
Repository dedicated to developing and testing privacy legal and compliance scoring of AI models.


Below is a concise summary of what this script does, its purpose, and what it hopes to demonstrate:

**Problem to be Solved**

- Software and analytics businesses seek to run machine learning and business analytics on usage data in order to improve their products and services, but usage data almost always contains personal data.
- This exposes businesses to potential regulatory fines, contractual violations, reputational harm, and increased data breach risk, and exposes individuals to unnecessary risk.
- No method of personal data is likely to be "perfect", thus a method of benchmarking filtering and redaction by LLM technology is needed.

**Proposed Solution**

- Develop a testing method for benchmarking performance in personal data redaction.
- Develop and report methods on small LLM performance.
- Identify LLM models that are small enough to be cheaply deployed locally in order to filter personal data out of usage or business data without the risks of cloud deployment, international data transfer, or other forms of breach, surveillance, or interception.

**Context of Problem**

- Privacy laws have broad definitions of personal data that indicate that any information that relates to an identified or identifiable living human is personal data.
- Traditional regex-based pattern searching is insufficiently flexible to understand the nuances.
- Modern LLM technology offers a potential avenue for identifying and protecting personal data.
- Businesses must be cost-aware in their deployment of privacy enhancing technology.

**What the script does**

- Loads a set of questions from a JSON file.
- Runs multiple AI models against these questions.
- Uses a "judging model" to grade the AI responses. (in this case gpt-4o)
- Converts the grades to numeric scores for ease of summary.
- Aggregates and compiles scores into a final report, along with a brief summary of strengths and weaknesses of each model in this task.

**Purpose**

- To automate testing, grading, and reporting for various AI models on privacy-related questions.
- To generate consistent performance metrics and a final summary of each model’s strengths and weaknesses.

**What it hopes to demonstrate**

- How to orchestrate a pipeline that tests multiple AI models.
- How to use a specialized model (like GPT-4) to grade and evaluate responses.
- How to systematically gather and present results, from raw responses all the way to final scores and summary insights.

