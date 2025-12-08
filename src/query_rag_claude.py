import os
import chromadb
from dotenv import load_dotenv
from anthropic import Anthropic
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load API key from .env
load_dotenv()

class ClaudeRAG:
    """Query RAG using Claude API for high-quality responses."""
    
    def __init__(self):
        print("Loading RAG system with Claude...")
        
        # Initialize Claude API
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Set up embedding model (same as used for indexing)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
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
        
        print("✅ RAG system loaded with Claude API!\n")
    
    def query(self, question: str, similarity_top_k: int = 5):
        """Query the RAG system using Claude."""
        # Retrieve MORE relevant documents (increased from 3 to 5)
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve(question)
        
        # Build context from retrieved documents with source info
        context_parts = []
        for i, node in enumerate(nodes, 1):
            source = node.metadata.get('filename', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{node.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create improved prompt for Claude
        prompt = f"""You are answering questions about someone's professional background based on their personal documents.

Retrieved documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information in the retrieved documents above
- If the documents don't contain relevant information, say so clearly
- Cite which source(s) you're using in your answer
- Be specific and detailed when information is available"""
        
        # Call Claude API
        message = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Print which sources were retrieved (for debugging)
        print("\n[Retrieved sources:]")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node.metadata.get('filename', 'Unknown')}")
        print()
        
        return message.content[0].text

if __name__ == "__main__":
    rag = ClaudeRAG()
    
    # Interactive query loop
    print("="*60)
    print("Professional RAG Query System (Powered by Claude)")
    print("="*60)
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        question = input("Question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if question.strip():
            print("\nThinking...\n")
            try:
                response = rag.query(question)
                print(f"Answer: {response}\n")
                print("-"*60 + "\n")
            except Exception as e:
                print(f"Error: {str(e)}\n")