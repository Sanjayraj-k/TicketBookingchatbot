from flask import Flask, request, jsonify
from flask_cors import CORS
import requests  # Importing the missing requests module
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os
from PyPDF2 import PdfReader

# API Key for OpenRouteService
ORS_API_KEY = "5b3ce3597851110001cf6248c1cce33a2c1f487bbb59575f02854d69"

# Initialize the LLM and Vector Database
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)

# Museum Coordinates (Chennai Museum)
MUSEUM_COORDINATES = {"lon": 80.2574, "lat": 13.0674}

# Store user sessions (for handling location-based queries)
user_sessions = {}

# Load and Split Documents
def load_texts(text_folder: str):
    documents = []
    for filename in os.listdir(text_folder):
        file_path = os.path.join(text_folder, filename)

        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith(".pdf"):
            pdf_reader = PdfReader(file_path)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            documents.append(Document(page_content=text, metadata={"source": filename}))

    if not documents:
        raise ValueError(f"No documents found in the folder: {text_folder}")
    return documents

text_folder = r"D:\llm1\pa"
docs = load_texts(text_folder)
print(f"Loaded {len(docs)} documents from {text_folder}.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

if not all_splits:
    raise ValueError("Document splitting failed. Ensure documents contain content.")
print(f"Split into {len(all_splits)} chunks.")

valid_splits = [doc for doc in all_splits if doc.page_content.strip()]
if not valid_splits:
    raise ValueError("No valid document chunks found after splitting.")

vector_store.add_documents(documents=valid_splits)
print("Document chunks added to vector store successfully.")

# Load RAG Prompt
prompt = hub.pull("rlm/rag-prompt")

# Define State for RAG Model
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# RAG Retrieval and Generation Functions
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Define RAG Model Graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})

# 🔹 Function to Get Coordinates from Location Name
def get_coordinates(location_name):
    url = f"https://api.openrouteservice.org/geocode/search?api_key={ORS_API_KEY}&text={location_name}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None  # Handle request failure
    
    data = response.json()

    try:
        coordinates = data["features"][0]["geometry"]["coordinates"]
        return {"lon": coordinates[0], "lat": coordinates[1]}
    except (KeyError, IndexError):
        return None

# 🔹 Function to Calculate Distance from User's Location to Chennai Museum
def get_distance(from_coords, to_coords):
    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {
        "locations": [
            [from_coords["lon"], from_coords["lat"]],
            [to_coords["lon"], to_coords["lat"]]
        ],
        "metrics": ["distance"]
    }

    response = requests.post(url, json=body, headers=headers)

    if response.status_code != 200:
        return "Sorry, I couldn't fetch the distance."

    data = response.json()

    try:
        distance_km = data["distances"][0][1] / 1000  # Convert meters to km
        return f"The distance from {from_coords['name']} to Chennai Museum is {distance_km:.2f} km."
    except KeyError:
        return "Sorry, I couldn't fetch the distance."

@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        # 🔹 Handle ticket booking separately
        ticket_keywords = [
            "book tickets", "book a ticket", "book ticket", "ticket booking", 
            "buy tickets", "purchase ticket", "get ticket", "reserve ticket",
            "museum entry ticket", "entry pass"
        ]
        
        if any(keyword in question.lower() for keyword in ticket_keywords):
            return jsonify({"answer": "Click here to <a href='http://localhost:4010' target='_blank'>book your tickets</a>."})

        # 🔹 Handle location-based distance queries
        session_id = request.remote_addr  # Use IP as session identifier

        distance_keywords = ["how far", "distance to", "reach", "far to", "from", "near", "travel to", "go to"]
        if session_id in user_sessions and user_sessions[session_id].get("waiting_for_location"):
            from_location = question.title()
            from_coords = get_coordinates(from_location)

            if not from_coords:
                return jsonify({"answer": f"Sorry, I couldn't find the location: {from_location}."})

            from_coords["name"] = from_location
            answer = get_distance(from_coords, MUSEUM_COORDINATES)

            # Clear session after answering the distance question
            del user_sessions[session_id]
            return jsonify({"answer": answer})

        if any(keyword in question.lower() for keyword in distance_keywords):
            user_sessions[session_id] = {"waiting_for_location": True}
            return jsonify({"answer": "Enter your current location to calculate the distance."})

        # 🔹 Process other questions using RAG model
        response = graph.invoke({"question": question})
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
