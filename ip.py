from pathlib import Path

from langchain_docling.loader import DoclingLoader, ExportType


# def load_documents(docs_path="docs"):
#     """
#     Load PDF, PPTX and TXT documents from the docs directory
#     using Docling and return LangChain Document objects.
#     """

#     docs_dir = Path(docs_path)

#     # -----------------------------------
#     # 1. Check directory
#     # -----------------------------------

#     if not docs_dir.exists():
#         raise FileNotFoundError(
#             f"Directory '{docs_path}' does not exist."
#         )

#     # -----------------------------------
#     # 2. Find supported files
#     # -----------------------------------

#     supported_extensions = {
#         ".pdf",
#         ".pptx",
#         ".txt"
#     }

#     files = [
#         file
#         for file in docs_dir.iterdir()
#         if file.is_file()
#         and file.suffix.lower() in supported_extensions
#     ]

#     if not files:
#         raise FileNotFoundError(
#             f"No supported documents found in '{docs_path}'."
#         )

#     print(f"Found {len(files)} documents.")

#     for file in files:
#         print(f"  - {file.name}")

#     # -----------------------------------
#     # 3. Load documents using Docling
#     # -----------------------------------

#     documents = []

#     for file in files:

#         print(f"\nProcessing: {file.name}")

#         loader = DoclingLoader(
#             file_path=str(file),
#             export_type=ExportType.MARKDOWN 
#         )

#         docs = loader.load()

#         documents.extend(docs)

#         print(
#             f"  Loaded {len(docs)} LangChain document(s)"
#         )

#     # -----------------------------------
#     # 4. Check result
#     # -----------------------------------

#     if not documents:
#         raise ValueError(
#             "Docling did not produce any documents."
#         )

#     print(
#         f"\nTotal LangChain documents created: "
#         f"{len(documents)}"
#     )

#     # -----------------------------------
#     # 5. Inspect first few documents
#     # -----------------------------------

#     for i, doc in enumerate(documents[:3]):

#         print(f"\n========== Document {i + 1} ==========")

#         print(
#             "Source:",
#             doc.metadata.get("source")
#         )

#         print(
#             "Content length:",
#             len(doc.page_content)
#         )

#         print(
#             "Content preview:",
#             doc.page_content[:500]
#         )

#         print(
#             "Metadata:",
#             doc.metadata
#         )

#     return documents



directory = Path("docs")
file_paths = [
    str(p) for ext in ("*.pdf", "*.pptx", "*.txt") for p in directory.glob(ext)
]

loader = DoclingLoader(file_path=file_paths,export_type= ExportType.MARKDOWN)
documents = loader.load()
print(f"number of docs {len(documents)}")

for i, doc in enumerate(documents[:3]):

        print(f"\n========== Document {i + 1} ==========")

        print(
            "Source:",
            doc.metadata.get("source")
        )

        print(
            "Content length:",
            len(doc.page_content)
        )

        print(
            "Content preview:",
            doc.page_content[:500]
        )

        print( "Metadata:")
        print(f"len = {len(doc.metadata)}")
        for k,v in list(doc.metadata.items())[:5]:
            print(k,v)



# def main():

#     print("Starting Docling ingestion...\n")

#     documents = load_documents("docs")

#     print("\nIngestion completed successfully.")


# if __name__ == "__main__":
#     main()