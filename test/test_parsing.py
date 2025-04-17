from parser import parse_tables

def test_docx_parsing():
    tables = parse_tables("data/sample_files/example.docx")
    assert isinstance(tables, list)
    assert all(isinstance(t, list) for t in tables)

def test_pdf_parsing():
    tables = parse_tables("data/sample_files/example.pdf")
    assert isinstance(tables, list)
    assert all(isinstance(t, list) for t in tables)
