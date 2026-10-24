# RAG Chatbot on Your Own Data

Chat with your PDFs and webpages using retrieval-augmented generation (RAG). The app
retrieves relevant passages from your documents and generates answers with source
citations and PDF page references.

Built as a local learning and portfolio project with **Python, LangChain, Groq,
Streamlit, ChromaDB, and Sentence Transformers**.

## Features

- Upload text-based PDFs or import public webpages.
- Combine semantic search and keyword matching to find relevant passages.
- Ask follow-up questions, review quotes, and preview cited PDF pages in the app.
- Search, inspect, and manage sources from the sidebar.
- Save named conversations, export chats, and keep your library between restarts.

## Setup

Requires **Python 3.11–3.13** and a [Groq API key](https://console.groq.com/keys)
for chat. Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Create a `.env` file from `.env.example` if you do not already have one, then set:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
RAG_DATA_DIR=data
```

Keep your real API key in `.env`; leave `.env.example` free of credentials.

Start the app:

```bash
streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). The first indexing operation downloads
the embedding model. After it is cached, indexing and search run locally; chat
requires internet access to Groq. Restart the app after editing Python files.

## Usage

1. Open **Add sources** and upload a PDF or enter a webpage URL.
2. Use **Chat** to ask questions and expand citations to review the evidence.
3. Use **Search passages** and **Inspect documents** to explore your sources.
4. Find sources in **Manage sources** to chat with, preview, inspect, or remove them.
5. In a citation card, click **Preview PDF page** to view that page and navigate the PDF.

Documents are split into chunks, embedded locally, and stored in ChromaDB. Each
question retrieves relevant chunks before Groq generates and checks the answer.
Questions, retrieved passages, and limited follow-up context are sent to Groq.
Uploaded files, model caches, and the local index stay in the ignored `data/` folder.
Use **Save chat** to name a conversation; later replies save automatically. Reopen it
under **Saved conversations** in the sidebar. Saved chats keep up to 50 turns and
their cited excerpts in `data/conversations.sqlite3`. Unnamed chats are session-only.

## Tests

Run the tests that do not need an API key or model download:

```bash
python -m pytest -m 'not integration and not live' -q
python -m ruff check .
```

The project also includes real-model integration tests and a synthetic evaluation
set. See the [evaluation and quality report](docs/quality-improvements.md) for details.

## Limitations

Scanned PDFs need OCR, which is not included. Complex tables and layouts may extract
poorly. Web imports support a single public HTML page, without login or JavaScript
rendering. Citation checks reduce errors, but answers can still omit details.
The app currently uses one shared local library and has no user authentication.

See the [architecture](docs/architecture.md), [demo guide](docs/demo.md), and
[deployment notes](docs/deployment.md) for more information. `requirements.lock.txt`
records the tested macOS / Python 3.13 environment.
