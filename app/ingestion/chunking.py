from langchain_text_splitters import RecursiveCharacterTextSplitter
import logfire

def chunk(doc):
    with logfire.span("chunking"):
        logfire.info("Chunking the document")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        chunked_doc = splitter.split_documents(doc)
        logfire.info("Chunked the document")
        return chunked_doc