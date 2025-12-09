# Professional RAG System

A Retrieval-Augmented Generation (RAG) system built to query my professional documents using semantic search and AI. This project demonstrates expertise in vector databases, embeddings, LLM integration, and cloud/local hybrid architecture.

## 🎯 Project Overview

This RAG system indexes my professional documents (resumes, projects, code samples, transcripts) and enables natural language queries about my background and experience. It combines local embedding generation with Claude AI for high-quality, context-aware responses.

## 🛠️ Tech Stack

**Core Technologies:**
- **LlamaIndex** - RAG framework and orchestration
- **ChromaDB** - Vector database for semantic search
- **Claude 3.5 Haiku API** - LLM for response generation
- **HuggingFace Embeddings** - Local embedding model (BAAI/bge-small-en-v1.5)

**Data Processing:**
- **Python 3.13** - Primary language
- **pypdf, python-docx, openpyxl, python-pptx** - Multi-format document parsing
- **Pandas & NumPy** - Data manipulation

**Infrastructure:**
- **Google Colab** - GPU-accelerated embedding generation
- **Local execution** - Cost-optimized query processing
- **Hybrid cloud/local architecture** - Balance performance and cost

## 🏗️ Architecture
```
┌─────────────────┐
│  Professional   │
│   Documents     │ (PDF, DOCX, XLSX, PY, SQL, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loader │ (Multi-format parsing)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │ (HuggingFace BGE-small)
│  Generation     │ [Run on Google Colab GPU]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │ (Vector storage - persistent)
│  Vector Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Search │ (Top-K retrieval)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Claude API     │ (Response synthesis)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AI Response    │
└─────────────────┘
```

## 📁 Project Structure
```
professional-rag/
├── data/
│   └── raw/              # Professional documents (not in Git)
├── src/
│   ├── document_loader.py    # Multi-format document parser
│   ├── rag_system.py         # Original RAG with Ollama
│   ├── query_rag_claude.py   # Production RAG with Claude
│   ├── rebuild_index.py      # Index rebuilding utility
│   └── check_index.py        # Index inspection tool
├── chroma_db/            # Vector database (local, not in Git)
├── .env                  # API keys (not in Git)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.13+
- Anthropic API key (for Claude)
- Google account (for Colab GPU indexing)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/professional-rag.git
cd professional-rag
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_api_key_here
```

4. **Add your documents:**
Place your professional documents in `data/raw/` directory.

### Building the Index (Google Colab)

For initial indexing or when adding many documents:

1. Open `Professional_RAG_Indexing.ipynb` in Google Colab
2. Upload your documents from `data/raw/`
3. Run all cells to generate embeddings (GPU-accelerated)
4. Download the `chroma_db.zip` file
5. Extract to project root directory

### Querying the System

Run the interactive query interface:
```bash
python src/query_rag_claude.py
```

Example queries:
- "Summarize my resume and professional experience"
- "What Python skill have I demonstrated?"
- "What was my GPA in college?"
### Running the Demo

To see a curated demonstration of the RAG system's capabilities:
```bash
python demo.py
```

This runs 3 pre-defined queries showcasing:
- Technical skills extraction
- Professional experience summarization
- Educational background

## 💡 Key Features

### Multi-Format Document Support
Supports PDF, Word, Excel, PowerPoint, Python, SQL, and text files with intelligent parsing.

### Hybrid Architecture
- **Cloud indexing:** GPU-accelerated embedding generation in Colab
- **Local querying:** Cost-effective execution on personal hardware
- **Total cost:** ~$0.001 per query with Claude Haiku

### Semantic Search
Uses state-of-the-art embeddings for accurate document retrieval based on meaning, not just keywords.

### Source Attribution
Responses cite which documents were used, ensuring transparency and verifiability.

## 📊 Performance

- **Indexed documents:** 15 professional documents
- **Total chunks:** 241 semantic chunks
- **Embedding model:** BAAI/bge-small-en-v1.5 (133M parameters)
- **Query latency:** ~2-3 seconds end-to-end
- **Cost per query:** <$0.001 USD

## 🔧 Maintenance

### Adding New Documents

**Option 1 - Full Rebuild (for many documents):**
1. Add documents to `data/raw/`
2. Run indexing in Colab (GPU-accelerated)
3. Download and replace `chroma_db/`

**Option 2 - Local Rebuild (for few documents):**
```bash
python src/rebuild_index.py
```

### Inspecting the Index
```bash
python src/check_index.py
```

## 🎓 Skills Demonstrated

- **Vector Databases:** ChromaDB implementation and optimization
- **Natural Language Processing:** Semantic search and embeddings
- **API Integration:** Anthropic Claude API, HuggingFace models
- **Cloud Computing:** Google Colab GPU utilization
- **Cost Optimization:** Hybrid cloud/local architecture
- **Python Development:** Multi-format parsing, async operations
- **MLOps Concepts:** Model selection, inference optimization


## 🤝 Contact

**Your Name**
- GitHub: [@Chenzoh12](https://github.com/Chenzoh12)
- LinkedIn: [Vincent Scotti](https://www.linkedin.com/in/scottivincent/)
- Email: ScottiVincentJ@gmail.com

## 📄 License

This project is for portfolio demonstration purposes.