from pathlib import Path

from langchain_docling.loader import DoclingLoader, ExportType
import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma 
from dotenv import load_dotenv 
from langchain_huggingface import HuggingFaceEmbeddings 
 

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

def create_vector_store(chunks, persist_directory="db/chroma_db"): 
    """Create and persist ChromaDB vector store""" 
    print("Creating embeddings and storing in ChromaDB...") 
         
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
 
    # Create ChromaDB vector store 
    print("--- Creating vector store ---") 
 
    vectorstore = Chroma.from_documents( 
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory,  
        collection_metadata={"hnsw:space": "cosine"} 
    ) 
    print("--- Finished creating vector store ---") 
     
    print(f"Vector store created and saved to {persist_directory}") 
    return vectorstore 


def split_documents(documents, chunk_size=1000, chunk_overlap=0): 
    """Split documents into smaller chunks with overlap""" 
    print("Splitting documents into chunks...") 
     
    text_splitter = CharacterTextSplitter( 
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap 
    ) 
     
    chunks = text_splitter.split_documents(documents) 
     
     
    return chunks 

def load_documents():

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

    return documents


def main(): 
    """Main ingestion pipeline""" 
    print("=== RAG Document Ingestion Pipeline ===\n") 
     
    # Define paths 
    docs_path = "docs" 
    persistent_directory = "db/chroma_db" 
     
    # Check if vector store already exists 
    if os.path.exists(persistent_directory): 
        print("✅ Vector store already exists. No need to re-process documents.") 
         
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma( 
            persist_directory=persistent_directory, 
            embedding_function=embedding_model,  
            collection_metadata={"hnsw:space": "cosine"} 
        ) 
        
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents") 
        return vectorstore 
 
     
    print("Persistent directory does not exist. Initializing vector store...\n") 
     
    # Step 1: Load documents 
    documents = load_documents(docs_path)   
 
    # Step 2: Split into chunks 
    chunks = split_documents(documents) 
     
    # # Step 3: Create vector store 
    vectorstore = create_vector_store(chunks, persistent_directory) 
     
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.") 
    return vectorstore 
 
 
if __name__ == "__main__": 
    main()