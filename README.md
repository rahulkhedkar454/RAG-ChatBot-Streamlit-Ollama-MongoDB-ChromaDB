Universal AI Assistant
A powerful, all-in-one AI assistant built to help with coding, data science, machine learning, and general knowledge queries. Powered by Ollama, MongoDB, and ChromaDB, it features a modern Streamlit interface, persistent vector memory, and multiple expert modes for specialized assistance.
Table of Contents

Setup Instructions
Project Flow
System Requirements
How It Works
Session Management
Vector Database Usage


Functions
Technologies Used
Understanding the Code

Setup Instructions
Prerequisites

Python: Version 3.8 or higher.
Ollama: Local server for AI model inference.
MongoDB: Local or remote instance for chat history storage.
ChromaDB: For persistent vector storage of embeddings.
uv: Python package manager for dependency management.

Step 1: Install uv
Install the uv package manager to manage Python dependencies:
curl -LsSf https://astral.sh/uv/install.sh | sh

Step 2: Set Up Ollama

Install Ollama:curl -fsSL https://ollama.com/install.sh | sh


Start the Ollama server:ollama serve


Pull required models:ollama pull phi3:mini
ollama pull nomic-embed-text



Step 3: Set Up MongoDB

Install MongoDB locally or use a cloud instance (e.g., MongoDB Atlas).
Ensure MongoDB is running:mongod


Configure the MongoDB URI in environment variables (see Step 4).

Step 4: Set Up the Project

Clone the repository:git clone <repository-url>
cd universal-ai-assistant


Create a virtual environment with uv:uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install dependencies using uv:uv pip install streamlit requests pymongo chromadb pandas plotly tiktoken tenacity ratelimit


Set environment variables in .env or directly in your shell:export MONGODB_URI="mongodb://localhost:27017/"
export OLLAMA_API_URL="http://localhost:11434"
export CHROMADB_PERSIST_DIR="./chromadb_data"

Alternatively, configure Streamlit secrets in .streamlit/secrets.toml:MONGODB_URI = "mongodb://localhost:27017/"
OLLAMA_API_URL = "http://localhost:11434"
CHROMADB_PERSIST_DIR = "./chromadb_data"



Step 5: Run the Application
uv run streamlit run app.py

The app will be available at http://localhost:8501.
Project Flow
The Universal AI Assistant follows a straightforward workflow:

User Input: Users enter queries via the Streamlit interface (e.g., code questions, ML tasks, or general queries).
Processing: Queries are sanitized and sent to the Ollama server, which processes them using the phi3:mini model (or other selected models).
Context Management: Relevant conversation history (from MongoDB) and learned memories (from ChromaDB) are retrieved to provide context-aware responses.
Response Generation: Ollama generates a response, which is streamed or displayed in the UI.
Storage: User queries and responses are saved to MongoDB for history and ChromaDB for persistent vector memory.
Display: The Streamlit UI renders the conversation, with formatted code blocks and markdown support.

System Requirements

Operating Systems: Windows, macOS, Linux.
Hardware:
Minimum: 4GB RAM, 2-core CPU, 10GB free disk space.
Recommended: 8GB RAM, 4-core CPU, SSD for faster storage.


Software:
Python 3.8 or higher.
MongoDB 4.4 or higher.
Ollama (latest version recommended).
uv package manager.


Network: Stable internet for initial model downloads; local network for Ollama API (default: http://localhost:11434).
Optional: Docker for containerized deployment of MongoDB or Ollama.

How It Works
The Universal AI Assistant is a web-based application built with Streamlit, leveraging Ollama for AI responses, MongoDB for chat history, and ChromaDB for persistent vector memory. Below is a detailed explanation of its core components and additional details on session management and vector database usage.
Session Management
Sessions are managed to maintain conversation continuity and allow users to revisit past interactions. The application uses a combination of Streamlit’s session_state and MongoDB to handle sessions effectively:

Session Creation: When a user starts a new chat or clears the current one, a unique session_id is generated using uuid.uuid4(). This ID is stored in st.session_state.session_id and used to track all messages in the session.
Storage in MongoDB: Each user query and assistant response is saved to MongoDB in the ai_assistant database, chat_history collection. The document includes:
session_id: Links messages to a specific session.
timestamp: When the interaction occurred.
user_message: The user’s query.
assistant_response: The assistant’s response.
prompt_type: The selected assistant mode (e.g., python_dev, general).
model: The Ollama model used (e.g., phi3:mini).
message_count: Tracks the number of messages in the session.
title: A generated title for new sessions, created using generate_session_title by summarizing the first few user messages.


Session Retrieval: Users can view recent sessions in the sidebar, where each session is listed with a preview of the first message and message count. Clicking a session loads its history from MongoDB using load_session_history, populating st.session_state.messages for display.
Session Summarization: If a session exceeds 15 messages (SUMMARIZE_THRESHOLD), users can trigger summarize_conversation to condense older messages into a summary, preserving context while reducing token usage.
Cleanup: The MongoDB connection is closed using cleanup_mongodb when a new session is started, the current chat is cleared, or the application exits (via atexit.register(cleanup_mongodb)).
UI Integration: The sidebar provides options to start a new session, load recent sessions, or clear the current chat, ensuring seamless session management.

Vector Database Usage
The application uses ChromaDB as a persistent vector database to store and retrieve text embeddings, enabling context-aware responses and persistent memory for learned content:

Initialization: The init_chromadb function sets up a persistent ChromaDB client with storage in the ./chromadb_data directory (configurable via CHROMADB_PERSIST_DIR). It creates or retrieves a collection named ai_assistant_memory using the OllamaEmbeddingFunction (or a custom fallback) for generating embeddings with the nomic-embed-text model.
Embedding Generation: The generate_embedding function sends text to the Ollama API (nomic-embed-text) to create vector embeddings, which are numerical representations of text for similarity searches.
Storing Embeddings: The save_embedding_to_vector_db function saves user inputs (e.g., from #learn commands) as embeddings in ChromaDB, along with metadata like session_id and timestamp. Each embedding is assigned a unique ID (e.g., <session_id>_<timestamp>).
Context Retrieval: The retrieve_similar_context function queries ChromaDB for the top 3 most similar embeddings to a user’s query, using cosine similarity. Retrieved documents are included in the prompt as [LEARNED MEMORY] to provide relevant context.
Persistence: Embeddings are stored persistently in the ./chromadb_data directory, ensuring that learned knowledge and context are retained across application restarts.
#learn Command: When a user enters a query starting with #learn (e.g., #learn Python decorators), the text is extracted, embedded, and stored in ChromaDB with save_embedding_to_vector_db. The assistant confirms with "✅ Learned and saved to memory." These learned items are retrieved later to enhance responses when similar queries are made.
Context Integration: Before sending a query to Ollama, the build_context function combines recent messages (from MongoDB) with similar memories (from ChromaDB) to create a context-rich prompt, ensuring responses are informed by both recent conversations and learned knowledge.

Application Workflow

Frontend: Streamlit provides a responsive UI with chat input, history display, and sidebar controls for model selection, context management, and session handling.
Backend:
Ollama: Handles natural language processing and embedding generation using models like phi3:mini and nomic-embed-text.
MongoDB: Stores chat history with session IDs, timestamps, and prompt types for persistence and retrieval.
ChromaDB: Maintains a persistent vector database for storing and querying text embeddings, enabling context-aware responses.


Data Flow:
User inputs are sanitized to prevent injection attacks (sanitize_input).
Queries are augmented with context from recent messages (MongoDB) and similar memories (ChromaDB).
Ollama processes the query and streams or returns a response (call_ollama_stream or call_ollama).
Responses are formatted (e.g., code blocks, markdown) and displayed in the UI.
Both query and response are saved to MongoDB (save_chat_to_db) and, if a #learn command, to ChromaDB.


Features:
Multiple assistant modes (e.g., Python dev, ML engineer, general).
Real-time streaming responses.
Session management with history and summarization.
Markdown export for conversations.
Persistent vector memory for learned content via #learn.



Functions
Key functions in the application include:

init_chromadb(): Initializes the persistent ChromaDB client and collection for vector storage.
init_mongodb(): Sets up the MongoDB connection for chat history.
cleanup_mongodb(): Closes the MongoDB connection on session end or process exit.
sanitize_input(): Prevents injection attacks by cleaning user input.
generate_embedding(): Creates text embeddings using Ollama’s nomic-embed-text model.
save_embedding_to_vector_db(): Stores embeddings in ChromaDB for memory retrieval, used for #learn commands.
retrieve_similar_context(): Queries ChromaDB for relevant context based on embeddings to enhance responses.
call_ollama_stream(): Streams responses from Ollama for real-time interaction.
call_ollama(): Handles non-streaming Ollama API calls.
build_context(): Constructs conversation context within token limits, combining MongoDB and ChromaDB data.
summarize_conversation(): Summarizes long conversations to maintain context.
save_chat_to_db(): Saves user queries and responses to MongoDB for session history.
load_session_history(): Retrieves chat history for a session from MongoDB.
get_recent_conversations(): Fetches recent session metadata for display in the sidebar.
load_all_sessions_with_messages(): Loads all sessions with their messages for history viewing.
generate_session_title(): Generates descriptive titles for new sessions.
generate_markdown_export(): Exports conversations as markdown files.

Technologies Used

Streamlit: Frontend framework for the web interface.
Ollama: AI model inference and embedding generation.
MongoDB: Database for storing chat history and session metadata.
ChromaDB: Persistent vector database for text embeddings.
Python Packages:
uv: Package manager for dependency installation.
requests: For API calls to Ollama.
pymongo: MongoDB client for Python.
chromadb: Vector database for persistent embeddings.
pandas, plotly: For statistics visualization.
tiktoken: For token estimation.
tenacity, ratelimit: For retry logic and rate limiting.


Python: Version 3.8+ for core application logic.

Understanding the Code
Project Structure

app.py: Main application file containing all logic, UI, and backend integration.
.streamlit/secrets.toml: Optional configuration for sensitive variables (e.g., MONGODB_URI).
chromadb_data/: Directory for persistent ChromaDB storage.

Key Code Components

Initialization: init_chromadb and init_mongodb use @st.cache_resource to ensure singletons for ChromaDB and MongoDB connections.
Session Management: Uses Streamlit’s session_state to track messages and session IDs, with MongoDB for persistence. Sessions are created, saved, and retrieved seamlessly.
Context Handling: Combines recent messages (MongoDB) and vector-based memories (ChromaDB) for context-aware responses. The #learn command enhances memory by storing user-specified knowledge.
Error Handling: Includes retries (tenacity), rate limiting (ratelimit), and input sanitization for robustness.
UI: Custom CSS for a modern look, with formatted code blocks and markdown rendering.

Extending the Code

Add New Assistant Modes: Modify PROMPT_TEMPLATES to include new expert roles.
Enhance Vector DB: Add cleanup functions for old embeddings or support for remote ChromaDB instances.
Improve UI: Add features like real-time token usage or advanced search for chat history.
Deployment: Use Docker to containerize the app, MongoDB, and Ollama for easier scaling.

Debugging Tips

Check the sidebar for connection statuses (Ollama, MongoDB, ChromaDB) and ChromaDB version.
Verify environment variables (MONGODB_URI, OLLAMA_API_URL, CHROMADB_PERSIST_DIR).
Monitor the ChromaDB persistence directory (./chromadb_data) for storage issues.
Use uv pip list to ensure all dependencies are installed correctly.
Test #learn functionality by saving knowledge and checking if it’s retrieved in relevant queries.

For further details, refer to the inline comments and docstrings in app.py.