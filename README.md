# 🧠 Prompt Evolution Engine

This project demonstrates a fully functional, self-optimizing prompt system using an LLM (GPT-4o/3.5) to evolve and test its own prompts, finding the optimal version based on measurable metrics.

## ✨ Features

* **Prompt Evolution:** An LLM generates multiple distinct variations of a user's base prompt.
* **Metric-Based Evaluation:** Each prompt variation is tested and scored against a set of quality metrics (e.g., Relevance, Readability, Length Score).
* **Leaderboard:** Displays a ranked table of all prompt versions and their scores.
* **Visualization (Plotly):** Provides a visual comparison of prompt performance across all metrics.
* **Evolution Loop:** The winning prompt can be automatically set as the new "Base Prompt" to start the next generation of evolution.
* **Report Download:** Saves a comprehensive log of the entire test run for analysis.

## 🚀 Setup & Installation

Follow these steps to set up and run the application locally.

### 1. Set Up the Project

Ensure you are in the main project directory (`Prompt_Engine`).

1.  **Activate Virtual Environment:** If it is not already active, run the activation script:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies:** Install all required Python libraries (Streamlit, OpenAI, Pandas, Plotly, etc.) using the `requirements.txt` file you generated:
    ```powershell
    py -m pip install -r requirements.txt
    ```

### 2. Set Your OpenAI API Key (Crucial)

The application requires your OpenAI API Key to function. You must set this key as an environment variable in your terminal before running the app.

*Replace `sk-xxxxxxxxxxxxxxxxxxxxxxxx` with your actual key.*
```powershell
$env:OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxx'
