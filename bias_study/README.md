# Bias_Study.py

Bias Study, is a Python script designed to generate summaries of news articles using multiple language models and then rate those summaries for bias using various criteria. The script interacts with several APIs (Gemini, Anthropic, OpenAI, Fireworks.AI (for DeepSeek and LLama 3.3) and Grok) and uses an input JSON file to drive its operations.

## Features

- **Multi-Model Summarization**: Generates summaries of a provided news article using a range of language models.
- **Leverages Cross-Model Correction**: Uses an average of all models to judge all other models.
- **Bias Rating**: Rates the generated summaries of source articles against predefined bias criteria.
- **Error Handling & Logging**: Implements error handling with logging for API calls and file operations.
- **Configurable**: Easily adjust models, bias rating criteria, and API keys via environment variables and the `.env` file.

## Requirements

- Python 3.7+
- Dependencies:
  - `python-dotenv`
  - `google-generativeai`
  - `anthropic`
  - `openai`
  - `requests`
  - `logging` (standard library)
  - `google-api-core` (for Google API exceptions)
  - Other standard libraries (`json`, `time`, `os`, `pathlib`)

Install required packages using pip:

```bash
pip install python-dotenv google-generativeai anthropic openai requests google-api-core
```

## Setup

1. **Clone the Repository**:  
   Clone or download the repository containing this script.

2. **Configure Environment Variables**:  
   Create a `.env` file (or update the provided one) with your API keys. Example:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   FW_DEEPSEEK_API_KEY=your_deepseek_api_key
   GROK_API_KEY=your_grok_api_key
   ```

3. **Set File Paths**:  
   The script uses a hard-coded path to the `.env` file. Modify the variable `env_path` if necessary:
   ```python
   env_path = r"C:<path>\PrivacyBench\.env"
   ```
   Adjust the path as needed for your environment.

4. **Prepare Input File**:  
   Create an `input.json` file in the same directory with the following structure:
   ```json
   {
       "article": "[Full text of the news article...]",
       "summary_prompt": "Summarize this article in 5 sentences.",
       "article_url": "http://www.newssite.com/articleurl"
   }
   ```
(you will need to use a JSON script to convert the article text to a JSON string)

## Configuration

- **Models for Summarization & Bias Rating**:  
  The script uses two dictionaries, `LLM_MODELS_SUMMARIZE` and `LLM_MODELS_JUDGE`, to define the language models and their associated APIs.
  
- **Bias Rating Criteria**:  
  The `BIAS_RATING_CRITERIA` list contains various bias dimensions. Modify or add criteria as needed.

- **Output File**:  
  Results are saved in `output_results.json`. This file will be updated after each run.

## Usage

Run the script from the command line:

```bash
python your_script_name.py
```

The script will:

1. Load the `.env` file and configure API keys.
2. Load the input JSON file containing the article and summary prompt.
3. Generate summaries using the models specified in `LLM_MODELS_SUMMARIZE`.
4. Rate the summaries for bias using the models in `LLM_MODELS_JUDGE` for each criterion defined.
5. Save the intermediate and final results to `output_results.json`.

## Troubleshooting

- **Missing `.env` File**:  
  Ensure the `.env` file exists at the specified path. The script prints a message if the file is not found.
  
- **API Errors**:  
  If API calls fail, check the logs for detailed error messages. Logging is configured to capture errors and warnings.
  
- **JSON Errors**:  
  Verify that your `input.json` file is correctly formatted and includes both `article` and `summary_prompt` keys.

- **Network Issues**:  
  The script includes error handling for network timeouts and request exceptions. If you encounter these issues, check your network connection and API endpoint availability.

## License

See repo license.  MIT open source license provided, attribution to Alex J. Wall, please and thank you.

---

For any issues or questions, please refer to the project documentation or contact the maintainers.