from langchain_community.document_loaders import PyPDFLoader
import logfire


def pdf_parser(file_path):
    with logfire.span("parsing"):
        logfire.info("Parsing the document")
        loader = PyPDFLoader(file_path)
        doc = loader.load()
        logfire.info("Parsed the document")
    return doc
