import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys
sys.path.append('src')
from document_loader import DocumentLoader

print("Setting up improved indexing...")

# Set up embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Better text splitter - smaller chunks with overlap
text_splitter = SentenceSplitter(
    chunk_size=512,  # Smaller chunks
    chunk_overlap=50  # Some overlap for context
)
Settings.text_splitter = text_splitter

# Delete old ChromaDB
import shutil
import os
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("✅ Deleted old index")

# Create new ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("professional_docs")
vector_store = ChromaVectorStore(chroma_collection=collection)

# Load documents
print("\nLoading documents...")
loader = DocumentLoader()
documents = loader.load_all_documents()

print(f"\n✅ Loaded {len(documents)} documents")
print("\nBuilding new index with better chunking...")

# Build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print("\n✅ Index rebuilt successfully!")
print(f"Total chunks created: {collection.count()}")