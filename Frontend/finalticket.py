from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import os
import logging
from PyPDF2 import PdfReader
import requests
import razorpay
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pymongo import MongoClient
import re
import uuid

# 🔥 Flask App Initialization
app = Flask(__name__)
app.secret_key = str(uuid.uuid4())  # Secure random secret key for session management
CORS(app, resources={r"/*": {"origins": "*"}})

# 🔥 Logging Configuration
logging.basicConfig(level=logging.INFO)

# 🔹 API Keys (Replace with your keys)
os.environ["LANGCHAIN_API_KEY"] = "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"  # Add your LangChain API key
os.environ["GROQ_API_KEY"] = "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"
PINECONE_API_KEY = "pcsk_2wjHhQ_ESQ6gNEBqgtpVHVwZYNoZNrtVpeBqw1Q5JrNtz6izYYUc6VwGVjUoytfFzr3H98"  # Replace with your Pinecone API key
PINECONE_INDEX_NAME = "museum-embeddings"
PINECONE_ENVIRONMENT = "us-west1-gcp"  # Replace with your Pinecone environment
ORS_API_KEY = "5b3ce3597851110001cf6248c1cce33a2c1f487bbb59575f02854d69"
RAZORPAY_KEY_ID = "rzp_test_1Ss2OE5DsbSMr0"
RAZORPAY_SECRET = "PSwn48wSWKAD0HwJptCOXoUt"

# 🔹 Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USERNAME = "ksanjayias@gmail.com"
EMAIL_PASSWORD = "hppd qdzf msvj hwlk"

# 🔹 MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["museum_db"]
bookings_collection = db["bookings"]

# 🔹 Razorpay Client
client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))

# 🔹 Language Models
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# 🔹 Museum Coordinates
MUSEUM_COORDINATES = {"lon": 80.2574, "lat": 13.0674}

# 🔥 User Session Storage
user_sessions = {}
# 🔥 Payment Storage (to track pending payments)
pending_payments = {}

# 🔹 Load Documents into Pinecone
def load_texts_to_pinecone(text_folder: str):
    documents = []
    for filename in os.listdir(text_folder):
        file_path = os.path.join(text_folder, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith(".pdf"):
            pdf_reader = PdfReader(file_path)
            text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            documents.append(Document(page_content=text, metadata={"source": filename}))

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    logging.info(f"Split {len(all_splits)} document chunks.")

    # Generate embeddings and upload to Pinecone
    for i, doc in enumerate(all_splits):
        embedding = embeddings.embed_query(doc.page_content)
        index.upsert(
            vectors=[{
                "id": f"doc_{i}_{str(uuid.uuid4())}",
                "values": embedding,
                "metadata": {"text": doc.page_content, "source": doc.metadata["source"]}
            }]
        )
    logging.info("Document embeddings uploaded to Pinecone successfully.")

# Load documents during app initialization
text_folder = "D:/llm1/pa"  # Update path for Render deployment
load_texts_to_pinecone(text_folder)

# 🔹 Define State for RAG Model
class State(Dict):
    question: str
    context: List[Document]
    answer: str

# 🔹 RAG Pipeline
def retrieve(state: State):
    question_embedding = embeddings.embed_query(state["question"])
    query_response = index.query(
        vector=question_embedding,
        top_k=4,
        include_metadata=True
    )
    retrieved_docs = [
        Document(
            page_content=match["metadata"]["text"],
            metadata={"source": match["metadata"]["source"]}
        )
        for match in query_response["matches"]
    ]
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 🔥 Graph Flow
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

# 🔹 Email Sending Function
def send_confirmation_email(email, name, tickets, date, payment_id):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USERNAME
        msg["To"] = email
        msg["Subject"] = "Museum Ticket Booking Confirmation"
        body = f"""
        Dear {name},
        Thank you for your payment! Your booking is confirmed for {tickets} tickets on {date}.
        Payment ID: {payment_id}
        Please bring this email or the payment ID on the day of your visit.
        Regards,
        Museum Team
        """
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, email, msg.as_string())
        logging.info(f"Confirmation email sent to {email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send confirmation email: {str(e)}")
        return False

# 🔹 Geocode Location Function
def geocode_location(location, api_key):
    url = f"https://api.openrouteservice.org/geocode/search?api_key={api_key}&text={location}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['features']:
            coordinates = data['features'][0]['geometry']['coordinates']
            return coordinates
    return None

# 🔹 Calculate Distance Function
def calculate_distance(start_lon, start_lat, end_lon, end_lat, api_key):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": api_key}
    body = {
        "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
        "units": "km"
    }
    response = requests.post(url, json=body, headers=headers)
    if response.status_code == 200:
        data = response.json()
        distance = data['routes'][0]['summary']['distance']
        return distance
    return None

# 🔹 Route: Home
@app.route('/')
def home():
    return "Welcome to the Museum Ticket Booking Chatbot!"

# 🔹 Route: Ask
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Invalid request. Missing 'question' parameter."}), 400

        question = data["question"].strip().lower()
        session_id = request.remote_addr

        # Handle distance query
        distance_match = re.search(r"i am (?:in )?(.+?) distance", question)
        if distance_match:
            location = distance_match.group(1).strip()
            start_coords = geocode_location(location, ORS_API_KEY)
            if start_coords:
                start_lon, start_lat = start_coords
                end_lon, end_lat = MUSEUM_COORDINATES["lon"], MUSEUM_COORDINATES["lat"]
                distance = calculate_distance(start_lon, start_lat, end_lon, end_lat, ORS_API_KEY)
                if distance:
                    return jsonify({"answer": f"The driving distance from {location.capitalize()} to the museum is approximately {distance:.2f} km."})
                else:
                    return jsonify({"answer": "Sorry, I couldn't calculate the distance."})
            else:
                return jsonify({"answer": f"Could not find the location '{location}'. Please provide a valid place."})

        # Step 1: Start Booking
        if "book ticket" in question:
            user_sessions[session_id] = {"step": "collect_details"}
            return jsonify({"answer": "Provide Name, Email, Tickets, and Date (YYYY-MM-DD), separated by commas."})

        # Step 2: Collect Booking Details
        if session_id in user_sessions:
            session = user_sessions[session_id]

            if session.get("step") == "collect_details":
                details = question.split(",")
                if len(details) != 4:
                    return jsonify({"answer": "Invalid format. Provide: Name, Email, Tickets, Date (YYYY-MM-DD)."})
                name, email, tickets, date = map(str.strip, details)
                try:
                    tickets = int(tickets)
                    if tickets <= 0:
                        raise ValueError
                except ValueError:
                    return jsonify({"answer": "Invalid number of tickets. Please provide a valid number."})
                amount = tickets * 5000
                session.update({
                    "name": name, "email": email, "tickets": tickets, "date": date, "amount": amount, "step": "confirm"
                })
                return jsonify({"answer": f"Confirm {tickets} tickets on {date} for {name} ({email})? Type 'yes' to proceed."})

            elif session.get("step") == "confirm" and question == "yes":
                payment_link = client.payment_link.create({
                    "amount": session["amount"] * 100,  # Razorpay expects amount in paise
                    "currency": "INR",
                    "accept_partial": False,
                    "description": "Museum Ticket Booking",
                    "customer": {
                        "name": session["name"],
                        "email": session["email"],
                        "contact": "+91XXXXXXXXXX"  # Replace with dynamic contact if available
                    },
                    "notify": {"sms": True, "email": True},
                    "reminder_enable": True,
                    "callback_url": request.url_root + "payment-callback",
                    "callback_method": "get"
                })
                payment_id = payment_link['id']
                payment_url = payment_link['short_url']
                pending_payments[payment_id] = {
                    "name": session["name"],
                    "email": session["email"],
                    "tickets": session["tickets"],
                    "date": session["date"],
                    "amount": session["amount"],
                    "status": "pending"
                }
                del user_sessions[session_id]
                return jsonify({
                    "answer": f"Please complete your payment by clicking <a href='{payment_url}' target='_blank'>here</a>. You will receive a confirmation email once payment is successful."
                })

        # RAG Response
        response = graph.invoke({"question": question})
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 🔹 Payment Callback Endpoint
@app.route('/payment-callback', methods=['GET', 'POST'])
def payment_callback():
    try:
        payment_id = request.args.get('razorpay_payment_link_id')
        payment_status = request.args.get('razorpay_payment_link_status')
        
        if payment_status == 'paid' and payment_id in pending_payments:
            booking = pending_payments[payment_id]
            
            # Send confirmation email
            send_confirmation_email(
                booking["email"],
                booking["name"],
                booking["tickets"],
                booking["date"],
                payment_id
            )
            
            # Store booking details in MongoDB
            booking_data = {
                "payment_id": payment_id,
                "name": booking["name"],
                "email": booking["email"],
                "tickets": int(booking["tickets"]),
                "date": booking["date"],
                "amount": booking["amount"],
                "status": "completed",
                "payment_date": "2025-06-19"  # Use current date or dynamic timestamp
            }
            bookings_collection.insert_one(booking_data)
            logging.info(f"Booking stored in MongoDB for {booking['email']}")
            
            # Update payment status
            pending_payments[payment_id]["status"] = "completed"
            
            return """
            <html>
                <head><title>Payment Successful</title></head>
                <body style="text-align: center; padding: 50px;">
                    <h1>Payment Successful!</h1>
                    <p>Your booking is confirmed. A confirmation email has been sent to your email address.</p>
                    <p>Thank you for booking with us!</p>
                    <a href="/">Return to Home</a>
                </body>
            </html>
            """
        
        return """
        <html>
            <head><title>Payment Status</title></head>
            <body style="text-align: center; padding: 50px;">
                <h1>Payment Not Completed</h1>
                <p>We couldn't verify your payment. Please try again or contact support.</p>
                <a href="/">Return to Home</a>
            </body>
        </html>
        """
        
    except Exception as e:
        logging.error(f"Payment callback error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 🔥 Run Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)