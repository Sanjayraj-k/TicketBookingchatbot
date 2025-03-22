from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
import razorpay
import logging
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})

# Razorpay API Keys (Replace with actual keys)
RAZORPAY_KEY_ID = "rzp_test_1Ss2OE5DsbSMr0"
RAZORPAY_KEY_SECRET = "PSwn48wSWKAD0HwJptCOXoUt"

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# In-memory session storage for user interactions
user_sessions = {}

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Database setup
def init_db():
    conn = sqlite3.connect("bookings.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        tickets INTEGER,
        date TEXT,
        payment_id TEXT
    )''')
    conn.commit()
    conn.close()

init_db()  # Initialize database

def save_booking(name, email, tickets, date, payment_id=None):
    """Saves booking details to the database."""
    conn = sqlite3.connect("bookings.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tickets (name, email, tickets, date, payment_id) VALUES (?, ?, ?, ?, ?)",
                   (name, email, tickets, date, payment_id))
    conn.commit()
    conn.close()

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip().lower()
        session_id = request.remote_addr  # Identify users via IP

        # Step 1: Start booking process
        ticket_keywords = ["book ticket", "ticket booking", "reserve ticket", "buy ticket"]
        if any(keyword in question for keyword in ticket_keywords):
            user_sessions[session_id] = {"step": "collect_details"}
            return jsonify({"answer": "Please provide your Name, Email, Number of Tickets, and Date (YYYY-MM-DD), separated by commas."})

        # Step 2: Collect booking details
        if session_id in user_sessions:
            session = user_sessions[session_id]

            if session.get("step") == "collect_details":
                details = question.split(",")  
                if len(details) != 4:
                    return jsonify({"answer": "Invalid format. Please provide: Name, Email, Number of Tickets, and Date."})

                name, email, tickets, date = [d.strip() for d in details]

                if not tickets.isdigit() or int(tickets) <= 0:
                    return jsonify({"answer": "Invalid ticket number. Please enter a valid number."})

                tickets = int(tickets)
                amount = tickets * 5000  # Convert to paise (₹50 per ticket)

                session.update({
                    "name": name,
                    "email": email,
                    "tickets": tickets,
                    "date": date,
                    "amount": amount,
                    "step": "confirm"
                })
                return jsonify({"answer": f"Confirm booking for {tickets} tickets on {date} for {name} ({email})? Type 'yes' to proceed with payment."})

            elif session.get("step") == "confirm" and question == "yes":
                booking = session
                try:
                    # Create a Razorpay payment link
                    payment_link = client.payment_link.create({
                        "amount": booking["amount"],
                        "currency": "INR",
                        "accept_partial": False,
                        "description": "Museum Ticket Booking",
                        "customer": {
                            "name": booking["name"],
                            "email": booking["email"],
                            "contact": "+919000090000"
                        },
                        "notify": {
                            "sms": True,
                            "email": True
                        },
                        "reminder_enable": True,
                        "callback_url": "https://example-callback-url.com/",
                        "callback_method": "get"
                    })

                    session["step"] = "payment"
                    session["payment_url"] = payment_link["short_url"]

                    return jsonify({
                        "answer": f"Click here to <a href='{payment_link['short_url']}' target='_blank'>book your tickets</a>."
})


                except Exception as e:
                    logging.error(f"Error creating Razorpay payment link: {str(e)}")
                    return jsonify({"answer": f"Payment setup failed: {str(e)}"})

        return jsonify({"answer": "I didn't understand your request. Please try again."})

    except Exception as e:
        logging.error(f"Error in /ask route: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/payment-success', methods=['POST'])
def payment_success():
    """Handles successful payments and updates the database."""
    try:
        data = request.get_json()
        payment_id = data.get("payment_id")

        if not payment_id:
            return jsonify({"message": "Payment failed."}), 400

        session_id = request.remote_addr
        if session_id in user_sessions:
            booking = user_sessions.pop(session_id)
            save_booking(booking["name"], booking["email"], booking["tickets"], booking["date"], payment_id)

            return jsonify({"message": "Payment successful! Your tickets are booked."})

        return jsonify({"message": "No booking found for this session."}), 404

    except Exception as e:
        logging.error(f"Error in /payment-success route: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
