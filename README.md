# 🤖 Chat with PDF locally using Ollama + LangChain

A powerful local RAG (Retrieval Augmented Generation) application that lets you chat with your PDF documents using Ollama and LangChain. This project includes a modern Next.js web app powered by a FastAPI backend, plus Jupyter notebooks for experimentation.

## ✨ Features

- 🔒 **100% Local** - All processing happens on your machine, no data leaves
- 📄 **Multi-PDF Support** - Upload and query across multiple documents
- 🧠 **Multi-Query RAG** - Intelligent retrieval with source citations
- 🎯 **Advanced RAG** - LangChain-powered pipeline with ChromaDB
- 🖥️ **Modern Next.js UI** - Chat interface with PDF management and source citations
- 🔌 **REST API** - FastAPI backend for programmatic access
- 📓 **Jupyter Notebooks** - For experimentation and learning

## 🖼️ Screenshots

### Next.js Interface
![Next.js UI](nextjs_ui.png)
*Modern chat interface with PDF management, source citations, and reasoning steps*

## 📺 Video Tutorial
<a href="https://youtu.be/ztBJqzBU5kc">
  <img src="https://img.youtube.com/vi/ztBJqzBU5kc/hqdefault.jpg" alt="Watch the video" width="100%">
</a>

## 🏗️ Project Structure
```
ollama_pdf_rag/
├── src/
│   ├── api/                  # FastAPI REST API
│   │   ├── routers/          # API endpoints
│   │   ├── services/         # Business logic
│   │   └── main.py           # API entry point
│   └── core/                 # Core RAG functionality
│       ├── document.py       # PDF processing
│       ├── embeddings.py     # Vector embeddings
│       ├── llm.py            # LLM configuration
│       └── rag.py            # RAG pipeline
├── web-ui/                   # Next.js frontend
│   ├── app/                  # Next.js app router
│   ├── components/           # React components
│   └── lib/                  # Utilities & AI integration
├── data/
│   ├── pdfs/                 # PDF storage
│   └── vectors/              # ChromaDB storage
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── run_api.py                # FastAPI runner (port 8001)
└── start_all.sh              # Start all services
```

## 🚀 Getting Started

### Prerequisites

1. **Install Ollama**
   - Visit [Ollama's website](https://ollama.ai) to download and install
   - Pull the required embedding and chat models:
     ```bash
     ollama pull nomic-embed-text  # Required for embeddings
     ollama pull phi4-mini         # Or: ollama pull llama3.2
     ```

2. **Clone Repository**
   ```bash
   git clone <your-repository-url>
   cd ollama_pdf_rag
   ```

3. **Set Up Python Environment**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

4. **Set Up Next.js Frontend**
   ```bash
   cd web-ui
   pnpm install  # or: npm install
   npx tsx lib/db/init-db.ts  # Initialize SQLite database
   cd ..
   ```

---

### 🎮 Running the Application

#### Step 1: Start the FastAPI Backend
Open a terminal in the project root:
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python run_api.py
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  python run_api.py
  ```
> Runs on: **http://localhost:8001**

#### Step 2: Start the Next.js Frontend
Open a second terminal:
```bash
cd web-ui
pnpm dev  # or: npm run dev
```
> Runs on: **http://localhost:3000**

#### Step 3: Access the Services
| Service | URL | Description |
|---|---|---|
| **Next.js Frontend** | [http://localhost:3000](http://localhost:3000) | Modern chat & PDF upload interface |
| **FastAPI Backend** | [http://localhost:8001](http://localhost:8001) | REST API |
| **API Documentation** | [http://localhost:8001/docs](http://localhost:8001/docs) | Swagger UI |
| **Ollama Server** | [http://localhost:11434](http://localhost:11434) | Local LLM & Embeddings service |

#### Option 2: Jupyter Notebook

```bash
jupyter notebook
```
Open `notebooks/experiments/updated_rag_notebook.ipynb` to experiment with the code.

## 💡 Usage

### Next.js Interface
1. **Upload PDFs** - Click the 📎 button or drag & drop files
2. **View PDFs** - Uploaded PDFs appear in the sidebar with chunk counts
3. **Select Model** - Choose from your locally available Ollama models
4. **Ask Questions** - Type your question and get answers with source citations
5. **View Reasoning** - See the AI's thinking process and retrieved chunks

## 🔌 API Reference

The FastAPI backend provides these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/pdfs/upload` | Upload and process a PDF |
| `GET` | `/api/v1/pdfs` | List all uploaded PDFs |
| `DELETE` | `/api/v1/pdfs/{pdf_id}` | Delete a PDF |
| `POST` | `/api/v1/query` | Query PDFs with RAG |
| `GET` | `/api/v1/models` | List available Ollama models |
| `GET` | `/api/v1/health` | Health check |

See full documentation at http://localhost:8001/docs when running.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## ⚠️ Troubleshooting

- **Ollama not responding**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Pull models with `ollama pull <model-name>`
- **No chunks retrieved**: Re-upload PDFs to rebuild the vector database
- **Port conflicts**: Check if ports 3000 or 8001 are in use

### Common Errors

#### ONNX DLL Error (Windows)
```
DLL load failed while importing onnx_copy2py_export
```
Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) and restart.

#### CPU-Only Systems
Reduce chunk size if experiencing memory issues:
- Modify `chunk_size` to 500-1000 in `src/core/document.py`

## 🤝 Contributing

- Open issues for bugs or suggestions
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.
