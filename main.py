import os
import uuid
import datetime
import streamlit as st
import openai
import pymongo
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ai_assistant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chat_history")

CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb_data")
CHROMADB_COLLECTION_NAME = os.getenv("CHROMCHADB_COLLECTION_NAME", "ai_assistant_memory")

MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 500))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 10))
SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", 15))

# Initialize MongoDB
def init_mongodb():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    return db[COLLECTION_NAME]

# Initialize ChromaDB
def init_chromadb():
    client = chromadb.PersistentClient(path=CHROMADB_PERSIST_DIR)
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-3-large"
    )
    collection = client.get_or_create_collection(
        name=CHROMADB_COLLECTION_NAME,
        embedding_function=embed_fn
    )
    return collection

# Generate embedding
def generate_embedding(text):
    emb = openai.Embedding.create(model="text-embedding-3-large", input=text)
    return emb["data"][0]["embedding"]

# Chat function using OpenAI
def call_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # you can change to gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses conversation history for better answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message["content"]

# Save to MongoDB
def save_chat_to_db(db, session_id, user_message, assistant_response):
    db.insert_one({
        "session_id": session_id,
        "timestamp": datetime.datetime.now(),
        "user_message": user_message,
        "assistant_response": assistant_response
    })

# Load session history
def load_session_history(db, session_id):
    return list(db.find({"session_id": session_id}).sort("timestamp", 1))

# Retrieve similar context from ChromaDB
def retrieve_similar_context(chroma_collection, query):
    results = chroma_collection.query(query_texts=[query], n_results=3)
    return results["documents"]

# Main App
def main():
    st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")
    st.title("💬 Conversational RAG Chatbot with Memory")

    # Initialize DBs
    mongo_collection = init_mongodb()
    chroma_collection = init_chromadb()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Sidebar options
    if st.sidebar.button("Start New Chat"):
        st.session_state.session_id = str(uuid.uuid4())

    # Input box
    user_message = st.text_input("Type your message:")
    if st.button("Send") and user_message.strip():
        # Get similar context
        similar_contexts = retrieve_similar_context(chroma_collection, user_message)
        context_text = "\n".join([doc[0] for doc in similar_contexts]) if similar_contexts else ""

        full_prompt = f"Context from previous chats:\n{context_text}\nUser: {user_message}"

        assistant_response = call_llm(full_prompt)

        # Save to MongoDB
        save_chat_to_db(mongo_collection, st.session_state.session_id, user_message, assistant_response)

        # Save learned memory to Chroma
        chroma_collection.add(
            documents=[user_message],
            embeddings=[generate_embedding(user_message)],
            ids=[str(uuid.uuid4())]
        )

    # Display history
    history = load_session_history(mongo_collection, st.session_state.session_id)
    for msg in history:
        st.markdown(f"**You:** {msg['user_message']}")
        st.markdown(f"**Assistant:** {msg['assistant_response']}")

if __name__ == "__main__":
    main()

