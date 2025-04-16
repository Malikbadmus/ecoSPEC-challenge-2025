from docx import Document
import pdfplumber
import os
import sys


input_file = sys.argv[1]


def parse_docx_tables(file_path):
    document = Document(file_path)
    tables = []

    for table in document.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)

    return tables

def parse_pdf_tables(file_path):
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                cleaned_table = [
                    [cell.strip() if cell else '' for cell in row]
                    for row in table
                ]
                tables.append(cleaned_table)

    return tables

def parse_tables(file_path):
    file_extention = os.path.splitext(file_path)[-1].lower()

    if file_extention == '.docx':
        return parse_docx_tables(file_path)
    elif file_extention == '.pdf':
        return parse_pdf_tables(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extention}")



if __name__ == "__main__":
    tables = parse_tables(input_file)


    for idx, table in enumerate(tables):
        print(f"\nTable {idx + 1}")
        for row in table:
            print(row)
