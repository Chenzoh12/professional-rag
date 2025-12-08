import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from document_loader import DocumentLoader

class ProfessionalRAG:
    """RAG system for professional documents using local models."""
    
    def __init__(self, collection_name: str = "professional_docs"):
        print("Initializing RAG system...")
        
        # Set up local embedding model (free)
        print("Loading embedding model...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Set up local LLM via Ollama (free)
        print("Connecting to Ollama...")

        #   Too big for local machin
        # Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

        #   New smaller model for local machine testing
        Settings.llm = Ollama(model="tinyllama", request_timeout=120.0)
        
        # Set up ChromaDB vector store (free, local)
        print("Setting up vector database...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        self.index = None
        print("RAG system initialized!\n")
    
    def build_index(self):
        """Load documents and build the vector index."""
        print("Loading documents from data/raw/...")
        loader = DocumentLoader()
        documents = loader.load_all_documents()
        
        if not documents:
            print("No documents found! Please add files to data/raw/")
            return
        
        print(f"\nBuilding index from {len(documents)} documents...")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("\nIndex built successfully!")
    
    def query(self, question: str, similarity_top_k: int = 3):
        """Query the RAG system."""
        if self.index is None:
            # Try to load existing index
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context
            )
        
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(question)
        
        return response

if __name__ == "__main__":
    # Example usage
    rag = ProfessionalRAG()
    
    # Build the index (only need to do this once, or when you add new documents)
    rag.build_index()
    
    # Example queries
    print("\n" + "="*50)
    print("RAG System Ready! Try some queries:")
    print("="*50 + "\n")
    
    test_queries = [
        "What are my main technical skills?",
        "Summarize my professional experience",
        "What Python projects have I worked on?"
    ]
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("-" * 50)
        response = rag.query(query)
        print(f"Answer: {response}\n")