# 🤖 NoCapGenAI

A powerful, all-in-one AI assistant designed to assist with coding, data science, machine learning, and general knowledge queries. Built with **Streamlit**, **Ollama**, **MongoDB**, and **ChromaDB**, it features a modern web interface, persistent vector memory for context-aware responses, and multiple expert modes for specialized assistance.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Project Flow](#project-flow)
- [System Requirements](#system-requirements)
- [How It Works](#how-it-works)
  - [Session Management](#session-management)
  - [Vector Database Usage](#vector-database-usage)
- [Functions](#functions)
- [Technologies Used](#technologies-used)
- [Understanding the Code](#understanding-the-code)

## Setup Instructions

### Prerequisites
- **Python**: Version 3.8 or higher.
- **Ollama**: Local server for AI model inference.
- **MongoDB**: Local or cloud-based instance for chat history storage.
- **ChromaDB**: For persistent vector storage of text embeddings.
- **uv**: Python package manager for streamlined dependency management.

### Step 1: Install uv
Install the `uv` package manager to handle Python dependencies efficiently:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Set Up Ollama
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Start the Ollama server in a terminal:
   ```bash
   ollama serve
   ```
3. Pull the required models:
   ```bash
   ollama pull phi3:mini
   ollama pull nomic-embed-text
   ```

### Step 3: Set Up MongoDB
1. Install MongoDB locally or use a cloud instance (e.g., MongoDB Atlas).
2. Ensure MongoDB is running:
   ```bash
   mongod
   ```
3. Configure the MongoDB URI in the `.env` file (see Step 4).

### Step 4: Set Up the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruvpatel004/RAG-ChatBot-Streamlit-Ollama-MongoDB-ChromaDB
   cd RAG-ChatBot-Streamlit-Ollama-MongoDB-ChromaDB
   ```
2. Create a virtual environment with `uv`:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies using `uv`:
   ```bash
   uv pip install streamlit requests pymongo chromadb pandas plotly tiktoken tenacity ratelimit
   ```
4. Create a `.env` file in the project root to manage environment variables:
   ```bash
   touch .env
   ```
   Add the following to `.env` (refer to `small.env` for reference):
   ```env
   # --- Model & API Configuration ---
   MODEL_NAME=phi3:mini
   OLLAMA_API_URL=http://localhost:11434

   # --- MongoDB Configuration ---
   MONGODB_URI=mongodb://localhost:27017/
   DATABASE_NAME=ai_assistant
   COLLECTION_NAME=chat_history

   # --- ChromaDB Configuration ---
   CHROMCHADB_COLLECTION_NAME=ai_assistant_memory
   CHROMADB_PERSIST_DIR=./chromadb_data

   # --- Context & Summary Settings ---
   MAX_CONTEXT_LENGTH=500
   CONTEXT_WINDOW=10
   SUMMARIZE_THRESHOLD=15

   # --- Rate Limiting ---
   CALLS=10
   PERIOD=60
   ```
   **Note**: The application uses `python-dotenv` to load variables from the `.env` file. Ensure the file is correctly formatted and placed in the project root. Alternatively, configure Streamlit secrets for deployment in `.streamlit/secrets.toml`:
   ```toml
   MODEL_NAME = "phi3:mini"
   OLLAMA_API_URL = "http://localhost:11434"
   MONGODB_URI = "mongodb://localhost:27017/"
   DATABASE_NAME = "ai_assistant"
   COLLECTION_NAME = "chat_history"
   CHROMCHADB_COLLECTION_NAME = "ai_assistant_memory"
   CHROMADB_PERSIST_DIR = "./chromadb_data"
   MAX_CONTEXT_LENGTH = 500
   CONTEXT_WINDOW = 10
   SUMMARIZE_THRESHOLD = 15
   CALLS = 10
   PERIOD = 60
   ```

### Step 5: Run the Application
Run the Streamlit application:
```bash
uv run streamlit run app.py
```

The app will be accessible at `http://localhost:8501` in your browser.

## Project Flow
The **🤖 NoCapGenAI** assistant follows a streamlined workflow:
1. **User Input**: Users submit queries (e.g., coding, machine learning, or general questions) through the Streamlit interface.
2. **Processing**: Queries are sanitized and sent to the Ollama server, processed using the `phi3:mini` model (or other selected models).
3. **Context Management**: Conversation history (from MongoDB) and learned memories (from ChromaDB) are retrieved to ensure context-aware responses.
4. **Response Generation**: Ollama generates responses, which are streamed or displayed in the UI.
5. **Storage**: Queries and responses are saved to MongoDB for history and to ChromaDB for persistent vector memory (via `#learn` commands).
6. **Display**: The Streamlit UI renders conversations with formatted code blocks and markdown support.

## System Requirements
- **Operating Systems**: Windows, macOS, Linux.
- **Hardware**:
  - Minimum: 4GB RAM, 2-core CPU, 10GB free disk space.
  - Recommended: 8GB RAM, 4-core CPU, SSD for faster storage.
- **Software**:
  - Python 3.8 or higher.
  - MongoDB 4.4 or higher.
  - Ollama (latest version recommended).
  - uv package manager.
- **Network**: Stable internet for initial model downloads; local network for Ollama API (default: `http://localhost:11434`).
- **Optional**: Docker for containerized deployment of MongoDB or Ollama.

## How It Works
**🤖 NoCapGenAI** is a web-based application powered by Streamlit, leveraging Ollama for AI responses, MongoDB for chat history, and ChromaDB for persistent vector memory. Below are details on its core components, session management, and vector database usage.

### Session Management
Sessions ensure conversation continuity and allow users to revisit past interactions:
- **Session Creation**: A unique `session_id` is generated using `uuid.uuid4()` when starting a new chat or clearing the current one, stored in `st.session_state.session_id`.
- **Storage in MongoDB**: Queries and responses are saved in the `ai_assistant` database, `chat_history` collection, with fields:
  - `session_id`: Links messages to a session.
  - `timestamp`: Records when the interaction occurred.
  - `user_message`: The user’s query.
  - `assistant_response`: The assistant’s response.
  - `prompt_type`: The assistant mode (e.g., `python_dev`, `general`).
  - `model`: The Ollama model used (e.g., `phi3:mini`).
  - `message_count`: Tracks the number of messages.
  - `title`: A generated title for new sessions, created via `generate_session_title`.
- **Session Retrieval**: Recent sessions are listed in the sidebar with a preview of the first message and message count. Selecting a session loads its history from MongoDB using `load_session_history`.
- **Session Summarization**: For sessions exceeding 15 messages (`SUMMARIZE_THRESHOLD`), `summarize_conversation` condenses older messages to preserve context and reduce token usage.
- **Cleanup**: The `cleanup_mongodb` function closes MongoDB connections on session changes or app exit (via `atexit.register(cleanup_mongodb)`).
- **UI Integration**: The sidebar offers options to start new sessions, load recent ones, or clear the current chat.

### Vector Database Usage
ChromaDB enables persistent vector storage for context-aware responses:
- **Initialization**: The `init_chromadb` function sets up a persistent ChromaDB client in `./chromadb_data` (configurable via `CHROMADB_PERSIST_DIR`), creating a collection named `ai_assistant_memory` using the `nomic-embed-text` model for embeddings.
- **Embedding Generation**: The `generate_embedding` function creates vector embeddings via Ollama’s `nomic-embed-text` model.
- **Storing Embeddings**: The `save_embedding_to_vector_db` function stores user inputs (e.g., from `#learn` commands) in ChromaDB with metadata like `session_id` and `timestamp`.
- **Context Retrieval**: The `retrieve_similar_context` function queries ChromaDB for the top 3 most similar embeddings to a user’s query, using cosine similarity, and includes them in prompts as `[LEARNED MEMORY]`.
- **Persistence**: Embeddings are saved in `./chromadb_data`, ensuring retention across restarts.
- **#learn Command**: Queries starting with `#learn` (e.g., `#learn Python decorators`) are embedded and stored in ChromaDB, with a confirmation message: "✅ Learned and saved to memory."
- **Context Integration**: The `build_context` function combines recent messages (MongoDB) and similar memories (ChromaDB) for context-rich prompts.

### Application Workflow
- **Frontend**: Streamlit provides a responsive UI with chat input, history display, and sidebar controls for model selection, context management, and session handling.
- **Backend**:
  - **Ollama**: Processes natural language and generates embeddings using `phi3:mini` and `nomic-embed-text`.
  - **MongoDB**: Stores chat history with session metadata.
  - **ChromaDB**: Manages persistent vector storage for embeddings.
- **Data Flow**:
  - Inputs are sanitized (`sanitize_input`) to prevent injection attacks.
  - Queries are augmented with context from MongoDB and ChromaDB.
  - Ollama processes queries via `call_ollama_stream` or `call_ollama`.
  - Responses are formatted (code blocks, markdown) and displayed.
  - Queries and responses are saved to MongoDB (`save_chat_to_db`) and, for `#learn` commands, to ChromaDB.
- **Features**:
  - Multiple assistant modes (e.g., Python dev, ML engineer, general).
  - Real-time streaming responses.
  - Session management with history and summarization.
  - Markdown export for conversations.
  - Persistent vector memory via `#learn`.

## Functions
Key functions include:
- **init_chromadb()**: Initializes ChromaDB client and collection.
- **init_mongodb()**: Sets up MongoDB connection.
- **cleanup_mongodb()**: Closes MongoDB connection on session end or exit.
- **sanitize_input()**: Cleans user input to prevent injection.
- **generate_embedding()**: Creates text embeddings with `nomic-embed-text`.
- **save_embedding_to_vector_db()**: Stores embeddings in ChromaDB for `#learn`.
- **retrieve_similar_context()**: Queries ChromaDB for relevant context.
- **call_ollama_stream()**: Streams Ollama responses.
- **call_ollama()**: Handles non-streaming Ollama API calls.
- **build_context()**: Combines MongoDB and ChromaDB data for prompts.
- **summarize_conversation()**: Summarizes long conversations.
- **save_chat_to_db()**: Saves queries and responses to MongoDB.
- **load_session_history()**: Retrieves session history from MongoDB.
- **get_recent_conversations()**: Fetches recent session metadata.
- **load_all_sessions_with_messages()**: Loads all sessions for history viewing.
- **generate_session_title()**: Creates descriptive session titles.
- **generate_markdown_export()**: Exports conversations as markdown.

## Technologies Used
- **Streamlit**: Web interface framework.
- **Ollama**: AI model inference and embedding generation.
- **MongoDB**: Database for chat history and session metadata.
- **ChromaDB**: Persistent vector database for text embeddings.
- **Python Packages**:
  - `uv`: Dependency management.
  - `requests`: Ollama API calls.
  - `pymongo`: MongoDB client.
  - `chromadb`: Vector database.
  - `pandas`, `plotly`: Statistics visualization.
  - `tiktoken`: Token estimation.
  - `tenacity`, `ratelimit`: Retry and rate-limiting logic.
  - `python-dotenv`: Loads `.env` file for environment variables.
- **Python**: Version 3.8+ for core logic.

## Understanding the Code
### Project Structure
- **app.py**: Main application file with UI and backend logic.
- **.env**: Stores environment variables (e.g., `MONGODB_URI`).
- **.streamlit/secrets.toml**: Optional configuration for deployment.
- **chromadb_data/**: Directory for ChromaDB storage.

### Key Code Components
- **Initialization**: `init_chromadb` and `init_mongodb` use `@st.cache_resource` for singleton connections.
- **Session Management**: Uses `st.session_state` and MongoDB for seamless session tracking.
- **Context Handling**: Combines MongoDB history and ChromaDB memories for context-aware responses.
- **Error Handling**: Includes retries (`tenacity`), rate limiting (`ratelimit`), and input sanitization.
- **UI**: Custom CSS ensures a modern look with formatted code and markdown rendering.

### Extending the Code
- **New Assistant Modes**: Add to `PROMPT_TEMPLATES` for new expert roles.
- **Vector DB Enhancements**: Implement cleanup for old embeddings or support remote ChromaDB.
- **UI Improvements**: Add real-time token counts or advanced history search.
- **Deployment**: Use Docker for containerized scaling of app, MongoDB, and Ollama.

### Debugging Tips
- Verify connection statuses (Ollama, MongoDB, ChromaDB) in the sidebar.
- Check `.env` file for correct `MONGODB_URI`, `OLLAMA_API_URL`, and `CHROMADB_PERSIST_DIR`.
- Monitor `./chromadb_data` for storage issues.
- Run `uv pip list` to confirm dependency installation.
- Test `#learn` by saving knowledge and verifying retrieval in similar queries.
- Ensure `python-dotenv` is installed if the app relies on `.env` file loading.

For further details, refer to inline comments and docstrings in `app.py`. Contributions and feedback are welcome at [https://github.com/Dhruvpatel004/RAG-ChatBot-Streamlit-Ollama-MongoDB-ChromaDB](https://github.com/Dhruvpatel004/RAG-ChatBot-Streamlit-Ollama-MongoDB-ChromaDB).
