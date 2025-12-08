import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

class QueryRAG:
    """Query existing RAG index without rebuilding."""
    
    def __init__(self):
        print("Loading RAG system...")
        
        # Set up embedding model (same as used for indexing)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Set up local LLM
        Settings.llm = Ollama(model="tinyllama", request_timeout=120.0)
        
        # Load existing ChromaDB
        print("Loading vector database...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_collection("professional_docs")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Load existing index
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context
        )
        
        print("✅ RAG system loaded!\n")
    
    def query(self, question: str, similarity_top_k: int = 3):
        """Query the RAG system."""
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(question)
        return response

if __name__ == "__main__":
    rag = QueryRAG()
    
    # Interactive query loop
    print("="*60)
    print("Professional RAG Query System")
    print("="*60)
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        question = input("Question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if question.strip():
            print("\nThinking...\n")
            response = rag.query(question)
            print(f"Answer: {response}\n")
            print("-"*60 + "\n")