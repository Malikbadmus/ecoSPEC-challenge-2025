# ecoSPEC-challenge-2025

This repo contains my solution to the ecoSPEC-challenge task 


## Parse Tables from Word and PDF

This tool extracts **all tables** from `.docx` and `.pdf` files and represents them as 2D Python lists.

### 📦 Requirements

- python-docx
- pdfplumber
- transformers
- fpdf
- prettytable
- accelerate

```bash
pip install -r requirements.txt
```

### Usage

```bash
run.sh [path_to_file]
```

## 🧠 GenAI Table Generator

This is a lightweight Python CLI tool that uses open-source language models (like Zephyr-7B) to generate *realistic tables* based on a prompt, and optionally exports them to DOCX or PDF.

### 🚀 Features

- Fill in AI-generated tables from user prompts
- Support for custom row and column headers
- Export tables to:
  - Console (plain text)
  - DOCX (Microsoft Word)
  - PDF (via docx2pdf)
- Uses open-source GenAI models (e.g., Zephyr-7B)

---

### ⚙ Installation

1. *Clone the repository* or copy the main.py file.

2. *Install dependencies*:

```bash
pip install transformers torch python-docx docx2pdf
```

### Usage

```bash
python main.py [OPTIONS]
```

### Examples

1. Generate a table and print to console
   
```bash
 python main.py --prompt "Programming languages comparison" \
  --rows Python Java C++ \
  --columns Speed Ease_of_Use Community \
  --output console
```

2. Generate and save as DOCX
   
```bash
python main.py --prompt "AI tools in 2025" \
  --rows ChatGPT Claude Gemini \
  --columns Speed Accuracy Cost \
  --output docx \
  --file ai_tools.docx
```

3. Generate and save as PDF
4. 
```bash
python main.py --prompt "Climate factors by continent" \
  --rows Africa Asia Europe \
  --columns Temperature Rainfall Humidity \
  --output pdf \
  --file climate_table.pdf
```

## 🧠 Model Used

- The tool uses HuggingFaceH4/zephyr-7b-beta by default. You can easily change it in main.py.
