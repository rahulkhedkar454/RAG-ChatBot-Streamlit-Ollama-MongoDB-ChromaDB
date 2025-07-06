import streamlit as st
import requests
import json
import time
import base64
import re
import os
import uuid
import atexit
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
import plotly.express as px
from typing import Iterator
import chromadb
from chromadb.utils import embedding_functions
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import tiktoken
# import pkg_resources

# Configuration
MODEL_NAME = "phi3:mini"
# OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = "http://localhost:11434"  # Default Ollama API URL
# MONGODB_URI = st.secrets.get("MONGODB_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "ai_assistant"
COLLECTION_NAME = "chat_history"
CHROMCHADB_COLLECTION_NAME = "ai_assistant_memory"
CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_data")

MAX_CONTEXT_LENGTH = 4000
CONTEXT_WINDOW = 10
SUMMARIZE_THRESHOLD = 15
CALLS = 10  # Rate limit: calls per minute
PERIOD = 60

# Custom Ollama Embedding Function
class OllamaEmbeddingFunction:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.api_url = f"{OLLAMA_API_URL}/api/embeddings"

    def __call__(self, texts: list) -> list:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model_name, "prompt": text},
                    timeout=10
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    st.warning(f"Failed to generate embedding for text: {text[:50]}...")
                    embeddings.append([])
            except Exception as e:
                st.warning(f"Embedding generation failed: {str(e)}")
                embeddings.append([])
        return embeddings

# Initialize ChromaDB client with persistent storage
@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB persistent client and collection."""
    try:
        os.makedirs(CHROMADB_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMADB_PERSIST_DIR)
        try:
            # Try using built-in OllamaEmbeddingFunction if available
            embedding_function = embedding_functions.OllamaEmbeddingFunction(
                model_name="nomic-embed-text",
                url=OLLAMA_API_URL
            )
        except AttributeError:
            # Fallback to custom embedding function
            embedding_function = OllamaEmbeddingFunction(model_name="nomic-embed-text")
        
        collection = client.get_or_create_collection(
            name=CHROMCHADB_COLLECTION_NAME,
            embedding_function=embedding_function
        )
        return collection
    except Exception as e:
        st.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
        return None

collection = init_chromadb()

# MongoDB connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection and return collection and client."""
    try:
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        return collection, client
    except ConnectionFailure:
        st.error("❌ MongoDB connection failed. Please ensure MongoDB is running.")
        return None, None

# def cleanup_mongodb():
#     """Close MongoDB connection."""
#     try:
#         if 'mongo_client' in st.session_state and st.session_state.mongo_client:
#             st.session_state.mongo_client.close()
#             del st.session_state.mongo_client
#     except Exception as e:
#         st.warning(f"Failed to cleanup MongoDB: {str(e)}")

# Register MongoDB cleanup on process exit
# atexit.register(cleanup_mongodb)

mongo_collection, mongo_client = init_mongodb()
if mongo_client:
    st.session_state.mongo_client = mongo_client

# Page configuration
st.set_page_config(
    page_title="🌟 Universal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .code-block {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Prompt templates
PROMPT_TEMPLATES = {
    "python_dev": """You are an expert Python developer and coding assistant. 
    You maintain context across our conversation and can reference previous code and discussions.
    Focus on writing clean, efficient, and well-documented Python code.
    Provide best practices, error handling, and optimization suggestions.
    
    Previous conversation context:
    {context}
    
    Current user question: {query}""",
    
    "ml_engineer": """You are an expert Machine Learning Engineer and Data Scientist.
    You remember our previous discussions about ML models, data, and experiments.
    Help with ML algorithms, data preprocessing, model training, evaluation, and deployment.
    Include code examples with popular libraries like scikit-learn, pandas, numpy, tensorflow, pytorch.
    
    Previous conversation context:
    {context}
    
    Current user question: {query}""",
    
    "mongodb_expert": """You are a MongoDB expert specializing in database design, queries, and optimization.
    You remember our previous discussions about database, queries, and performance.
    Provide efficient MongoDB queries, schema design advice, and performance optimization tips.
    Include Python code using pymongo when relevant.
    
    Previous conversation context:
    {context}
    
    Current user question: {query}""",
    
    "code_review": """You are a senior code reviewer with context of our ongoing code review session.
    You remember previous code snippets and review feedback we've discussed.
    Analyze the following code for:
    - Best practices
    - Performance optimization
    - Security issues
    - Code readability
    - Potential bugs
    
    Previous conversation context:
    {context}
    
    Code/Question: {query}""",
    
    "debug_helper": """You are a debugging expert who remembers our debugging session.
    You can reference previous error messages, code attempts, and solutions we've tried.
    Provide step-by-step debugging approach and suggest solutions.
    
    Previous conversation context:
    {context}
    
    Code/Error: {query}""",
    
    "general": """You are a helpful, knowledgeable, and creative general-purpose AI assistant.
    You can answer questions, provide explanations, help with coding, solve problems, brainstorm ideas, and assist with a wide range of topics including technology, science, history, language, and more.
    You maintain context of our conversation and can reference previous topics and code.
    Provide clear, practical, and insightful responses.
    
    Previous conversation context:
    {context}
    
    Current user question: {query}"""
}

# Utility functions
def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    return re.sub(r'[<>;{}]', '', text.strip())

def generate_embedding(text: str) -> list:
    """Generate text embedding using Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10
        )
        if response.status_code != 200:
            raise Exception(f"Error generating embedding: {response.text}")
        return response.json()["embedding"]
    except Exception as e:
        st.warning(f"Failed to generate embedding: {str(e)}")
        return []

def save_embedding_to_vector_db(text: str, session_id: str, metadata: dict = {}):
    """Save text and its embedding to ChromaDB."""
    if collection is None:
        st.error("❌ ChromaDB not initialized.")
        return
    try:
        vector = generate_embedding(text)
        if vector:
            collection.add(
                documents=[text],
                embeddings=[vector],
                ids=[f"{session_id}_{time.time()}"],
                metadatas=[metadata]
            )
    except Exception as e:
        st.warning(f"Failed to save to vector DB: {str(e)}")

def check_ollama_status() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except requests.RequestException:
        return []

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_ollama_stream(prompt: str, model: str = MODEL_NAME) -> Iterator[str]:
    """Calls the Ollama API with streaming enabled to generate responses incrementally.
    
    Args:
        prompt (str): The input prompt for the model.
        model (str): The name of the model to use (default: MODEL_NAME).
    
    Yields:
        str: Incremental response chunks from the model.
    
    Returns:
        str: The complete response string.
    """
    try:
        prompt = sanitize_input(prompt)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=300
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            full_response += data['response']
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response
        else:
            error_msg = f"Error: HTTP {response.status_code}"
            yield error_msg
            return error_msg
            
    except requests.RequestException as e:
        error_msg = f"Error connecting to Ollama: {str(e)}"
        yield error_msg
        return error_msg

def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    """Call Ollama API without streaming."""
    try:
        prompt = sanitize_input(prompt)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response received")
        return f"Error: HTTP {response.status_code} - {response.text}"
            
    except requests.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}"

def estimate_tokens(text: str) -> int:
    """Estimate token count for text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        return len(text) // 4

def build_context(messages: list, max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """Build context from recent messages, staying within token limit."""
    if not messages:
        return ""
    
    recent_messages = messages[-CONTEXT_WINDOW:]
    context_parts = []
    current_length = 0
    
    for message in reversed(recent_messages):
        role = message["role"]
        content = message["content"]
        
        formatted_msg = f"{'User' if role == 'user' else 'Assistant'}: {content}"
        msg_tokens = estimate_tokens(formatted_msg)
        
        if current_length + msg_tokens > max_length:
            break
            
        context_parts.insert(0, formatted_msg)
        current_length += msg_tokens
    
    return "\n\n".join(context_parts)

def summarize_conversation(messages: list, model: str = MODEL_NAME) -> list:
    """Summarize older parts of conversation to maintain context."""
    if len(messages) < SUMMARIZE_THRESHOLD:
        return messages
    
    recent_messages = messages[-CONTEXT_WINDOW:]
    older_messages = messages[:-CONTEXT_WINDOW]
    
    summary_text = ""
    for msg in older_messages:
        if msg["role"] == "user":
            summary_text += f"User asked: {msg['content'][:100]}...\n"
        else:
            summary_text += f"Assistant provided: {msg['content'][:100]}...\n"
    
    summary_prompt = f"""Please provide a concise summary of this conversation focusing on:
    - Key topics discussed
    - Code examples shared
    - Problems solved
    - Important context to remember
    
    Conversation to summarize:
    {summary_text}
    
    Summary:"""
    
    try:
        summary = call_ollama(summary_prompt, model)
        summary_message = {
            "role": "assistant", 
            "content": f"[CONVERSATION SUMMARY: {summary}]",
            "is_summary": True
        }
        return [summary_message] + recent_messages
    except:
        return recent_messages

def save_chat_to_db(user_message: str, assistant_response: str, prompt_type: str, session_id: str = None):
    """Save chat to MongoDB."""
    if mongo_collection is not None:
        try:
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            session_id = st.session_state.session_id

            is_new_session = not mongo_collection.find_one({"session_id": session_id})
            title = generate_session_title(st.session_state.messages) if is_new_session else None

            chat_entry = {
                "timestamp": datetime.now(),
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "prompt_type": prompt_type,
                "model": MODEL_NAME,
                "message_count": len(st.session_state.messages)
            }

            if title:
                chat_entry["title"] = title

            mongo_collection.insert_one(chat_entry)
        except Exception as e:
            st.error(f"Failed to save to database: {str(e)}")

def retrieve_similar_context(query: str, top_k: int = 3) -> list:
    """Retrieve similar context from ChromaDB."""
    if collection is None:
        st.error("❌ ChromaDB not initialized.")
        return []
    try:
        query_vector = generate_embedding(query)
        if query_vector:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            return results['documents'][0] if results and results['documents'] else []
        return []
    except Exception as e:
        st.warning(f"Memory retrieval failed: {e}")
        return []

def load_session_history(session_id: str, limit: int = 50) -> list:
    """Load chat history for a session."""
    if mongo_collection is not None:
        try:
            chats = list(mongo_collection.find({"session_id": session_id}).sort("timestamp", 1).limit(limit))
            return chats
        except Exception as e:
            st.error(f"Failed to load session history: {str(e)}")
            return []
    return []

def get_recent_conversations(limit: int = 10) -> list:
    """Get recent conversation sessions."""
    if mongo_collection is not None:
        try:
            pipeline = [
                {"$sort": {"timestamp": -1}},
                {"$group": {
                    "_id": "$session_id",
                    "latest_timestamp": {"$first": "$timestamp"},
                    "message_count": {"$sum": 1},
                    "first_message": {"$last": "$user_message"}
                }},
                {"$sort": {"latest_timestamp": -1}},
                {"$limit": limit}
            ]
            sessions = list(mongo_collection.aggregate(pipeline))
            return sessions
        except Exception as e:
            st.error(f"Failed to get recent conversations: {str(e)}")
            return []
    return []

def load_all_sessions_with_messages() -> list:
    """Load all sessions with their messages."""
    if mongo_collection is None:
        return []

    try:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$session_id",
                "latest_timestamp": {"$first": "$timestamp"}
            }},
            {"$sort": {"latest_timestamp": -1}}
        ]
        sorted_session_ids = [doc["_id"] for doc in mongo_collection.aggregate(pipeline)]
        all_sessions = []

        for session_id in sorted_session_ids:
            session_messages = list(
                mongo_collection.find({"session_id": session_id}).sort("timestamp", 1)
            )

            summary = None
            title = None
            messages = []

            for chat in session_messages:
                if chat.get("is_summary"):
                    summary = chat.get("content")
                elif chat.get("title") and not title:
                    title = chat.get("title")
                elif chat.get("user_message") and chat.get("assistant_response"):
                    messages.append({
                        "user": chat.get("user_message", ""),
                        "assistant": chat.get("assistant_response", "")
                    })

            if not title:
                title = messages[0]["user"][:50] if messages else "Untitled Session"

            all_sessions.append({
                "session_id": session_id,
                "title": title.strip(),
                "summary": summary,
                "messages": messages,
                "total_messages": len(messages),
                "timestamp": session_messages[0]["timestamp"] if session_messages else None
            })

        return all_sessions
    except Exception as e:
        st.error(f"Failed to load sessions: {str(e)}")
        return []

def generate_session_title(messages: list, model: str = MODEL_NAME) -> str:
    """Generate a short, meaningful title for the session."""
    if not messages:
        return "Untitled Session"

    context_snippet = "\n".join(
        f"User: {m['content']}" for m in messages[:3] if m["role"] == "user"
    )

    prompt = f"""
    Based on the following conversation, generate a short and descriptive title (max 10 words):
    {context_snippet}
    Title:"""

    try:
        title = call_ollama(prompt, model)
        return title.strip().replace('"', '')
    except:
        return "Untitled Session"

def get_chat_stats() -> tuple:
    """Get chat statistics."""
    if mongo_collection is not None:
        try:
            total_chats = mongo_collection.count_documents({})
            prompt_types = mongo_collection.aggregate([
                {"$group": {"_id": "$prompt_type", "count": {"$sum": 1}}}
            ])
            return total_chats, list(prompt_types)
        except Exception as e:
            st.error(f"Failed to get stats: {str(e)}")
            return 0, []
    return 0, []

def generate_markdown_export(messages: list, title: str = "Chat Session") -> str:
    """Generate Markdown export of chat session."""
    export_md = f"# {title}\n\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Model:** {MODEL_NAME}\n\n"
    for i, msg in enumerate(messages):
        export_md += f"**User {i+1}:**\n{msg['user']}\n\n"
        export_md += f"**Assistant:**\n{msg['assistant']}\n\n"
    return export_md

def export_as_markdown_button(messages: list, title: str):
    """Render a button to download chat as Markdown."""
    md_content = generate_markdown_export(messages, title)
    b64 = base64.b64encode(md_content.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{title}.md">📥 Download Chat as Markdown</a>'
    st.markdown(href, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Assistant Settings")
    
    ollama_status = check_ollama_status()
    if ollama_status:
        st.success("✅ Ollama Server Connected")
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox("Select Model", available_models,
                                        index=0 if MODEL_NAME not in available_models else available_models.index(MODEL_NAME))
            MODEL_NAME = selected_model
        else:
            st.warning("⚠️ No models found. Please pull some models first.")
            st.code("ollama pull phi3:mini")
    else:
        st.error("❌ Ollama Server Not Running")
        st.code("ollama serve")
    
    use_streaming = st.checkbox("Enable Streaming Response", value=True)
    prompt_type = st.selectbox("Assistant Type", [
        "python_dev", "ml_engineer", "mongodb_expert", 
        "code_review", "debug_helper", "general"
    ])
    
    st.header("🧠 Context Management")
    if st.session_state.get("messages"):
        total_messages = len(st.session_state.messages)
        context_length = estimate_tokens(build_context(st.session_state.messages))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", total_messages)
        with col2:
            st.metric("Context Tokens", context_length)
        
        if total_messages > SUMMARIZE_THRESHOLD:
            if st.button("📝 Summarize Conversation"):
                st.session_state.messages = summarize_conversation(st.session_state.messages, MODEL_NAME)
                st.success("Conversation summarized to maintain context!")
                st.rerun()
        
        context_window = st.slider("Context Window", 5, 20, CONTEXT_WINDOW)
        if context_window != CONTEXT_WINDOW:
            st.session_state.context_window = context_window
    
    st.header("💬 Session Management")
    if st.button("🆕 New Chat Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        # cleanup_mongodb()  # Clean up MongoDB connection on new session
        st.success("Started new chat session!")
        st.rerun()
    
    recent_sessions = get_recent_conversations(5)
    if recent_sessions:
        st.subheader("Recent Sessions")
        for session in recent_sessions:
            session_label = f"{session['first_message'][:30]}... ({session['message_count']} messages)"
            if st.button(session_label, key=f"session_{session['_id']}"):
                session_history = load_session_history(session['_id'])
                st.session_state.messages = []
                for chat in session_history:
                    st.session_state.messages.append({"role": "user", "content": chat["user_message"]})
                    st.session_state.messages.append({"role": "assistant", "content": chat["assistant_response"]})
                st.session_state.session_id = session['_id']
                st.success(f"Loaded session with {len(session_history)} messages!")
                st.rerun()
    
    st.header("⚡ Quick Actions")
    if st.button("📊 View Statistics"):
        st.session_state.show_stats = True
    
    if st.button("📚 Chat History"):
        st.session_state.show_history = True
    
    if st.button("🔄 Load All Sessions"):
        st.session_state.show_sessions_history = True

    if st.button("🗑️ Clear Current Chat"):
        st.session_state.messages = []
        # cleanup_mongodb()  # Clean up MongoDB connection on clear
        st.rerun()
    
    if st.button("🔄 Refresh Models"):
        st.rerun()
    
    st.header("🤖 Model Info")
    st.info(f"Current Model: {MODEL_NAME}")
    st.info(f"Assistant Type: {prompt_type.replace('_', ' ').title()}")
    st.info(f"Streaming: {'Enabled' if use_streaming else 'Disabled'}")
    
    st.header("🗄️ Database")
    if mongo_collection is not None:
        st.success("✅ MongoDB Connected")
    else:
        st.error("❌ MongoDB Disconnected")
    
    st.header("🧠 Vector Database")
    if collection is not None:
        st.success("✅ ChromaDB Connected")
        st.info(f"Persistence Directory: {CHROMADB_PERSIST_DIR}")
        # st.info(f"ChromaDB Version: {pkg_resources.get_distribution('chromadb').version}")
    else:
        st.error("❌ ChromaDB Disconnected")

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌟 Universal AI Assistant</h1>
    <p>Your all-in-one expert for coding, learning, brainstorming, and more — powered by Ollama</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show statistics
if st.session_state.get("show_stats", False):
    st.header("📊 Usage Statistics")
    total_chats, prompt_types = get_chat_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Conversations", total_chats)
    with col2:
        st.metric("Current Model", MODEL_NAME)
    with col3:
        st.metric("Database Status", "Connected" if mongo_collection else "Disconnected")
    
    if prompt_types:
        df = pd.DataFrame(prompt_types)
        fig = px.pie(df, values='count', names='_id', title='Prompt Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Hide Statistics"):
        st.session_state.show_stats = False
        st.rerun()

# Show all sessions
if st.session_state.get("show_sessions_history", False):
    st.session_state.all_sessions = load_all_sessions_with_messages()
    st.header("📚 All Chat Sessions")


    for session in st.session_state.all_sessions:
        title = session["title"]
        date = session["timestamp"].strftime('%Y-%m-%d %H:%M') if isinstance(session["timestamp"], datetime) else "Unknown Date"
        label = f"📌 {title} — {date} | {session['total_messages']} messages"

        with st.expander(label):
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("▶️ Load this session", key=f"load_{session['session_id']}"):
                    st.session_state.messages = []
                    for chat in session["messages"]:
                        st.session_state.messages.append({"role": "user", "content": chat["user"]})
                        st.session_state.messages.append({"role": "assistant", "content": chat["assistant"]})
                    st.session_state.session_id = session["session_id"]
                    st.success(f"Session '{title}' loaded!")
                    st.rerun()

            with col2:
                export_as_markdown_button(session["messages"], session["title"])

            if session["summary"]:
                st.markdown(f"📝 **Summary:** {session['summary']}")

            for i, msg in enumerate(session["messages"]):
                st.markdown(f"**User {i+1}:** {msg['user']}")
                st.markdown(f"**Assistant:** {msg['assistant']}")

# Show chat history
if st.session_state.get("show_history", False):
    st.header("📚 Recent Chat History")
    history = load_session_history(st.session_state.get("session_id", str(uuid.uuid4())), 20)
    
    if history:
        for chat in history:
            timestamp_str = chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(chat['timestamp'], datetime) else str(chat['timestamp'])
            with st.expander(f"Chat from {timestamp_str}"):
                st.markdown(f"**User:** {chat['user_message'][:100]}...")
                st.markdown(f"**Assistant:** {chat['assistant_response'][:200]}...")
                st.markdown(f"**Type:** {chat['prompt_type']} | **Model:** {chat['model']}")
    else:
        st.info("No chat history found.")
    
    if st.button("Hide History"):
        st.session_state.show_history = False
        st.rerun()

# Chat interface
st.header("💬 Chat Interface")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
        else:
            if "```" in content:
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        lang = part.split('\n')[0] if part.split('\n')[0] in ['python', 'sql', 'bash'] else 'text'
                        st.code('\n'.join(part.split('\n')[1:]) if lang != 'text' else part, language=lang)
                    else:
                        st.markdown(f'<div class="assistant-message">{part}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about Python, ML, or MongoDB..."):
    st.session_state.show_stats = False
    st.session_state.show_history = False
    st.session_state.show_sessions_history = False

    if not check_ollama_status():
        st.error("❌ Ollama server is not running. Please start it with `ollama serve`")
        st.stop()

    if prompt.lower().strip().startswith("#learn"):
        learn_text = prompt.replace("#learn", "").strip()
        if learn_text:
            session_id = st.session_state.get("session_id", str(uuid.uuid4()))
            st.session_state.session_id = session_id
            with st.spinner("Saving to memory..."):
                save_embedding_to_vector_db(
                    learn_text,
                    session_id=session_id,
                    metadata={"source": "user_memory", "timestamp": str(datetime.now())}
                )
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "✅ Learned and saved to memory."})

            with st.chat_message("user"):
                st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
            with st.chat_message("assistant"):
                st.markdown('<div class="assistant-message">✅ Learned and saved to memory.</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Nothing to learn. Please add text after `#learn`.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    base_context = build_context(st.session_state.messages)
    memory_snippets = retrieve_similar_context(prompt, top_k=3)
    memory_context = "\n".join(f"[LEARNED MEMORY]\n{m}" for m in memory_snippets if m)
    combined_context = memory_context + "\n\n" + base_context

    formatted_prompt = PROMPT_TEMPLATES[prompt_type].format(
        context=combined_context,
        query=prompt
    )

    with st.chat_message("assistant"):
        if use_streaming:
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("🤔 Thinking..."):
                for chunk in call_ollama_stream(formatted_prompt, MODEL_NAME):
                    if chunk and not chunk.startswith("Error"):
                        full_response += chunk
                        response_placeholder.markdown(
                            f'<div class="assistant-message">{full_response}▎</div>',
                            unsafe_allow_html=True
                        )
                response_placeholder.markdown(
                    f'<div class="assistant-message">{full_response}</div>',
                    unsafe_allow_html=True
                )
        else:
            with st.spinner("🤔 Thinking..."):
                full_response = call_ollama(formatted_prompt, MODEL_NAME)
                if "```" in full_response:
                    parts = full_response.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 1:
                            lang = part.split('\n')[0] if part.split('\n')[0] in ['python', 'sql', 'bash'] else 'text'
                            st.code('\n'.join(part.split('\n')[1:]) if lang != 'text' else part, language=lang)
                        else:
                            st.markdown(f'<div class="assistant-message">{part}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.spinner("Saving to database..."):
        save_chat_to_db(prompt, full_response, prompt_type)

# Example prompts
st.header("💡 Example Prompts")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Python Development")
    example_prompts = [
        "Create a REST API using FastAPI with authentication",
        "Implement a decorator for caching function results",
        "Write a class for handling CSV file operations",
        "Create a context manager for database connections"
    ]
    
    for example in example_prompts:
        if st.button(example, key=f"py_{example[:20]}"):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()

with col2:
    st.subheader("ML Engineering")
    ml_prompts = [
        "Create a machine learning pipeline for text classification",
        "Implement cross-validation for hyperparameter tuning",
        "Build a feature engineering pipeline for time series data",
        "Create a model evaluation framework with multiple metrics"
    ]
    
    for example in ml_prompts:
        if st.button(example, key=f"ml_{example[:20]}"):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🤖 Built with:**")
    st.markdown("• Streamlit")
    st.markdown("• Ollama API")
    st.markdown("• MongoDB")
    st.markdown("• ChromaDB")
    
with col2:
    st.markdown("**🌍 Perfect for:**")
    st.markdown("• Coding & Debugging")
    st.markdown("• Data Science & ML")
    st.markdown("• General Knowledge & Q&A")
    st.markdown("• Brainstorming & Ideas")
    
with col3:
    st.markdown("**✨ Features:**")
    st.markdown("• Real-time streaming replies")
    st.markdown("• Chat history & sessions")
    st.markdown("• Multiple expert modes")
    st.markdown("• Persistent vector memory")
    st.markdown("• Beautiful, modern UI")

st.markdown("---")
st.markdown("**Pro Tip:** Try different assistant types for specialized or general help! 💡")
