from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing import List, Dict
import os
import logging
from PyPDF2 import PdfReader
import requests
import razorpay
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 🔥 Flask App Initialization
app = Flask(__name__)
app.secret_key = "super_secret_key"  # For session management
CORS(app, resources={r"/*": {"origins": "*"}})

# 🔥 Logging Configuration
logging.basicConfig(level=logging.INFO)

# 🔹 API Keys (Replace with your keys)
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"
ORS_API_KEY = "5b3ce3597851110001cf6248c1cce33a2c1f487bbb59575f02854d69"
RAZORPAY_KEY_ID = "rzp_test_1Ss2OE5DsbSMr0"
RAZORPAY_SECRET = "PSwn48wSWKAD0HwJptCOXoUt"

# 🔹 Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USERNAME = "ksanjayias@gmail.com"
EMAIL_PASSWORD = "hppd qdzf msvj hwlk"

# 🔹 Razorpay Client
client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))

# 🔹 Language Models
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)

# 🔹 Museum Coordinates
MUSEUM_COORDINATES = {"lon": 80.2574, "lat": 13.0674}

# 🔥 User Session Storage
user_sessions = {}
# 🔥 Payment Storage (to track pending payments)
pending_payments = {}

# 🔹 Load Documents into Vector Store
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
            text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            documents.append(Document(page_content=text, metadata={"source": filename}))

    return documents

text_folder = "D:/llm1/pa"
docs = load_texts(text_folder)
logging.info(f"Loaded {len(docs)} documents from {text_folder}.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store.add_documents(documents=all_splits)
logging.info("Document chunks added to vector store successfully.")

prompt = hub.pull("rlm/rag-prompt")

# 🔹 Define State for RAG Model
class State(Dict):
    question: str
    context: List[Document]
    answer: str

# 🔹 RAG Pipeline
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 🔥 Graph Flow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
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

# 🔹 Route: Home
@app.route('/')
def home():
    return "Welcome to the Museum Ticket Booking Chatbot!"

# 🔹 Route: Ask
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # 🔥 Extract and validate JSON input
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Invalid request. Missing 'question' parameter."}), 400

        question = data["question"].strip().lower()
        session_id = request.remote_addr

        # 🔥 Step 1: Start Booking
        if "book ticket" in question:
            user_sessions[session_id] = {"step": "collect_details"}
            return jsonify({"answer": "Provide Name, Email, Tickets, and Date (YYYY-MM-DD), separated by commas."})

        # 🔥 Step 2: Collect Booking Details
        if session_id in user_sessions:
            session = user_sessions[session_id]

            if session.get("step") == "collect_details":
                details = question.split(",")
                if len(details) != 4:
                    return jsonify({"answer": "Invalid format. Provide: Name, Email, Tickets, Date (YYYY-MM-DD)."})

                name, email, tickets, date = map(str.strip, details)
                amount = int(tickets) * 5000  

                session.update({
                    "name": name, "email": email, "tickets": tickets, "date": date, "amount": amount, "step": "confirm"
                })

                return jsonify({"answer": f"Confirm {tickets} tickets on {date} for {name} ({email})? Type 'yes' to proceed."})

            elif session.get("step") == "confirm" and question == "yes":
                # Create payment link
                payment_link = client.payment_link.create({
                    "amount": session["amount"], 
                    "currency": "INR", 
                    "accept_partial": False,
                    "description": "Museum Ticket Booking",
                    "customer": {
                        "name": session["name"], 
                        "email": session["email"], 
                        "contact": "+91XXXXXXXXXX"
                    },
                    "notify": {"sms": True, "email": True}, 
                    "reminder_enable": True,
                    "callback_url": request.url_root + "payment-callback",  # Add callback URL
                    "callback_method": "get"
                })
                
                # Store payment details for callback
                payment_id = payment_link['id']
                payment_url = payment_link['short_url']
                
                # Store booking info in pending payments
                pending_payments[payment_id] = {
                    "name": session["name"],
                    "email": session["email"],
                    "tickets": session["tickets"],
                    "date": session["date"],
                    "amount": session["amount"],
                    "status": "pending"
                }
                
                # Clear chat session
                del user_sessions[session_id]
                
                # Return response with payment link
                return jsonify({
                    "answer": f"Please complete your payment by clicking <a href='{payment_url}' target='_blank'>here</a>. You will receive a confirmation email once payment is successful."
                })

        # 🔥 RAG Response
        response = graph.invoke({"question": question})
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 🔹 Payment Callback Endpoint
@app.route('/payment-callback', methods=['GET', 'POST'])
def payment_callback():
    try:
        # Get payment ID and status from the callback
        payment_id = request.args.get('razorpay_payment_link_id')
        payment_status = request.args.get('razorpay_payment_link_status')
        
        # Verify the payment
        if payment_status == 'paid' and payment_id in pending_payments:
            # Get booking details
            booking = pending_payments[payment_id]
            
            # Send confirmation email
            send_confirmation_email(
                booking["email"],
                booking["name"],
                booking["tickets"],
                booking["date"],
                payment_id
            )
            
            # Update payment status
            pending_payments[payment_id]["status"] = "completed"
            
            # Return success page
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
        
        # If payment failed or not found
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
    app.run(debug=True, port=5000)