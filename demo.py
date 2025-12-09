"""
Professional RAG System Demo
Demonstrates key capabilities with pre-defined queries
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')
from query_rag_claude import ClaudeRAG

load_dotenv()

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_query(question, response, sources):
    """Print formatted query and response"""
    print(f"❓ QUERY: {question}")
    print(f"\n📚 SOURCES USED:")
    for i, source in enumerate(sources, 1):
        print(f"   {i}. {source}")
    print(f"\n💡 RESPONSE:\n{response}")
    print("\n" + "-"*70)

def run_demo():
    """Run the demo with curated questions"""
    
    print_header("PROFESSIONAL RAG SYSTEM - LIVE DEMO")
    print("Initializing system...\n")
    
    # Initialize RAG
    rag = ClaudeRAG()
    
    # Define demo queries that showcase different aspects
    demo_queries = [
        {
            "category": "Technical Skills",
            "question": "Summarize my proficiency in Python and SQL."
        },
        {
            "category": "Professional Experience",
            "question": "Summarize my resume, professional experience and key projects I supported."
        },
        {
            "category": "Education & Credentials",
            "question": "What is my educational and certification background?"
        }
    ]
    
    # Run each demo query
    for i, item in enumerate(demo_queries, 1):
        print_header(f"DEMO {i}/3: {item['category']}")
        
        question = item['question']
        print(f"Querying: {question}\n")
        print("Retrieving relevant documents and generating response...\n")
        
        # Get retrieval info
        retriever = rag.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(question)
        sources = [node.metadata.get('filename', 'Unknown') for node in nodes]
        
        # Get response
        response = rag.query(question)
        
        print_query(question, response, sources)
        
        # Pause between queries for readability
        if i < len(demo_queries):
            input("\n[Press Enter to continue to next demo...]\n")
    
    # Final summary
    print_header("DEMO COMPLETE")
    print("This RAG system demonstrates:")
    print("  ✓ Multi-format document parsing (PDF, DOCX, XLSX, PY, SQL)")
    print("  ✓ Semantic search with vector embeddings")
    print("  ✓ Context-aware AI responses using Claude")
    print("  ✓ Source attribution and transparency")
    print("  ✓ Cost-optimized architecture (~$0.001 per query)")
    print("\nThank you for viewing this demo!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Please ensure:")
        print("  1. ANTHROPIC_API_KEY is set in .env file")
        print("  2. chroma_db folder exists in project root")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")