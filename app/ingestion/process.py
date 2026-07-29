import argparse
import sys
import os
import glob
from pathlib import Path
import logfire
from dotenv import load_dotenv

from .parse import pdf_parser
from .chunking import chunk
from .vectorDB import insert_documents, wipe_collection

load_dotenv()
logfire.configure()

def process_directory(data_dir: str):
    with logfire.span(f"Processing directory: {data_dir}"):
        if not os.path.isdir(data_dir):
            logfire.error(f"Directory not found: {data_dir}")
            print(f"Error: Directory '{data_dir}' not found.")
            sys.exit(1)
        
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        
        if not pdf_files:
            logfire.info("No PDF files found in the specified directory.")
            print("No PDF files found.")
            return

        for pdf_path in pdf_files:
            print(f"Processing {pdf_path}...")
            # 1. Parse PDF
            docs = pdf_parser(pdf_path)
            
            if docs:
                # 2. Chunk documents
                chunked_docs = chunk(docs)
                
                # 3. Insert into Vector Database
                if chunked_docs:
                    insert_documents(chunked_docs)
                    print(f"Successfully processed and stored {pdf_path}")
                else:
                    print(f"No chunks created from {pdf_path}")
            else:
                print(f"No content extracted from {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector database.")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the directory containing PDF files."
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Wipe the existing Qdrant collection before ingesting new data."
    )
    
    args = parser.parse_args()
    
    if args.wipe:
        print("Wiping existing Qdrant collection...")
        wipe_collection()
        print("Collection wiped.")
        
    print(f"Starting ingestion from directory: {args.data_dir}")
    process_directory(args.data_dir)
    print("Ingestion pipeline completed.")

if __name__ == "__main__":
    main()