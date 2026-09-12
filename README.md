# OS-HOPE

OS-HOPE is a human-oversight multi-agent assistant for operating-system automation using Large Language Models (LLMs).

Unlike fully autonomous OS agents, OS-HOPE separates **planning** from **execution**. The LLM can analyze the user's request, generate an execution plan, and propose OS commands, but it does not have direct authority to execute them. The user reviews and authorizes the proposed plan before execution.

The system supports both a **command-line interface (CLI)** and a **web interface**.

# Installation

## 1. Download the Repository

## 2. Create a Python Environment

OS-HOPE can be installed using either **Conda** or Python's built-in **venv**.

### Option 1: Conda

Create a new Conda environment:

```bash
conda create -n oshope python=3.9
```

Activate the environment:

```bash
conda activate oshope
```

### Option 2: Python venv

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 4. Install OS-HOPE

Install OS-HOPE as a Python package:

```bash
pip install .
```

For development, an editable installation can be used instead:

```bash
pip install -e .
```

## 5. Configure Environment Variables

OS-HOPE requires an API key for the configured LLM provider.

Add the required API key:

```bash
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your API key.

---

# Usage

After completing the installation and configuring the required environment variables, OS-HOPE can be used through either the command-line interface or the web interface.

## Command-Line Interface

Start the OS-HOPE CLI using:

```bash
oshope-cli
```

## Web Interface

Start the OS-HOPE web interface using:

```bash
oshope-web
```

Then open the following URL in your web browser:

```bash
http://localhost:8501
```