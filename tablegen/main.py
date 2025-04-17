import argparse
from typing import List, Callable, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from prettytable import PrettyTable
from docx import Document
from fpdf import FPDF
import torch
import os

def load_mistral_pipeline():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    print("Device set to use CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # Forces CPU usage
    )
    return pipe

def mistral_model_call(pipe, prompt: str) -> List[str]:
    output = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
    cleaned = output.replace(prompt, "").strip()
    values = [v.strip() for v in cleaned.split(",") if v.strip()]
    return values

def generate_table_structure(pipe, topic: str) -> Tuple[List[str], List[str]]:
    prompt_cols = f"Generate 3-5 column headers for a table on the topic: {topic}."
    cols = mistral_model_call(pipe, prompt_cols)

    prompt_rows = f"Generate 3-5 rows or items that should appear in a table on: {topic}."
    rows = mistral_model_call(pipe, prompt_rows)

    return cols, rows

def fill_table_rows(
    prompt: str,
    columns: List[str],
    rows: List[str],
    model_call_fn: Callable[[str], List[str]]
) -> List[List[str]]:
    full_table = []
    for row_label in rows:
        full_prompt = (
            f"{prompt}\n"
            f"Component: {row_label}\n"
            f"Columns: {', '.join(columns[1:])}\n"
            f"Provide your response as comma-separated values."
        )
        model_response = model_call_fn(full_prompt)
        full_row = [row_label] + model_response
        full_table.append(full_row)
    return full_table

def print_table(columns: List[str], rows: List[List[str]]):
    table = PrettyTable()
    table.field_names = columns
    for row in rows:
        if len(row) == len(columns):
            table.add_row(row)
        else:
            table.add_row(row + [""] * (len(columns) - len(row)))
    print(table)

def export_table_to_docx(columns: List[str], rows: List[List[str]], intro: str, filename="output.docx"):
    doc = Document()
    doc.add_paragraph(intro)
    table = doc.add_table(rows=1 + len(rows), cols=len(columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(columns):
        hdr_cells[i].text = col
    for row in rows:
        cells = table.add_row().cells
        for i, cell_val in enumerate(row):
            cells[i].text = cell_val
    doc.save(filename)
    print(f"DOCX saved to {filename}")

def export_table_to_pdf(columns: List[str], rows: List[List[str]], intro: str, filename="output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, intro)

    line_height = pdf.font_size * 2.5
    effective_page_width = pdf.w - 2 * pdf.l_margin
    col_width = effective_page_width / len(columns)

    pdf.set_font("Arial", size=12, style="B")
    for col in columns:
        pdf.cell(col_width, line_height, col, border=1)
    pdf.ln(line_height)

    pdf.set_font("Arial", size=12)
    for row in rows:
        for val in row:
            pdf.cell(col_width, line_height, val, border=1)
        pdf.ln(line_height)
    pdf.output(filename)
    print(f"PDF saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="GenAI Table Filler")
    parser.add_argument("--prompt", type=str, help="Prompt to guide model table filling", required=False)
    parser.add_argument("--topic", type=str, help="Topic to generate full table from scratch", required=False)
    parser.add_argument("--columns", type=str, nargs="+", help="Column headers (optional)")
    parser.add_argument("--rows", type=str, nargs="+", help="Row names (optional)")
    parser.add_argument("--output", type=str, choices=["console", "pdf", "docx"], default="console")
    parser.add_argument("--intro", type=str, default="This document contains an AI-generated table based on your input.")
    args = parser.parse_args()

    print("Loading model...")
    mistral_pipe = load_mistral_pipeline()

    if args.topic:
        columns, rows = generate_table_structure(mistral_pipe, args.topic)
        prompt = f"Using the topic '{args.topic}', fill in the table rows."
    else:
        prompt = args.prompt or "Fill in the table based on the given prompt and headers."
        columns = args.columns or ["Aspect", "Detail", "Reasoning"]
        rows = args.rows or ["Item 1", "Item 2", "Item 3"]

    print("Generating table content...")
    full_table = fill_table_rows(
        prompt=prompt,
        columns=columns,
        rows=rows,
        model_call_fn=lambda p: mistral_model_call(mistral_pipe, p)
    )

    if args.output == "pdf":
        export_table_to_pdf(columns, full_table, args.intro)
    elif args.output == "docx":
        export_table_to_docx(columns, full_table, args.intro)
    else:
        print("\nGenerated Table:\n")
        print_table(columns, full_table)

if __name__ == "__main__":
    main()
