// File: app.js

const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");

const app = express();
app.use(bodyParser.json());

// 🔑 Replace with your credentials
const VERIFY_TOKEN = "sanjay"; // This must match the one in your Meta App Dashboard
const WHATSAPP_TOKEN = "EAATCYRPgyw4BPum83zwJANfif90mxaYevZAxyKd0ZCDW7cmjOi1TfEKST1TT6YQEYhjDBSZBLBZATnk2CVCY1ipkFuS0Mju9BxWkEolOCOA4RasG5gPEZCR3iebTZBA8R3lAjOwc2lg8ZC2tw8WCs8c4bZBBovxmRPaBDvf5o4Rs1rFwMeRtoJkkKOO3uz4S7A096uKFZA1tt8oc1NH5lBbZBl7Q1zGbpdtbuBH7pbMWhqMAZDZD";
const PHONE_NUMBER_ID = "842332982291402";

// 🤖 Your Python Chatbot Backend URL
const PYTHON_API_URL = "http://localhost:5000/ask";

// ✅ Step 3: Webhook verification (for setup in Meta App Dashboard)
app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    console.log("✅ Webhook verified!");
    res.status(200).send(challenge);
  } else {
    console.error("❌ Webhook verification failed.");
    res.sendStatus(403);
  }
});

// ✅ Step 4: Handle incoming messages and forward to Python backend
app.post("/webhook", async (req, res) => {
  try {
    const entry = req.body.entry?.[0];
    const changes = entry?.changes?.[0]?.value?.messages?.[0];

    // Ensure it's a text message from a user
    if (changes && changes.type === "text") {
      const from = changes.from; // User's WhatsApp number (e.g., 919876543210)
      const msgBody = changes.text.body;

      console.log(`📩 Message from ${from}: "${msgBody}"`);

      // 🚀 FORWARD THE MESSAGE TO THE PYTHON BACKEND
      let botResponse = "Sorry, I'm having a little trouble right now. Please try again in a moment.";

      try {
        const pythonResponse = await axios.post(PYTHON_API_URL, {
          question: msgBody,
          session_id: from, // Use the user's phone number as the unique session ID
        });

        if (pythonResponse.data && pythonResponse.data.answer) {
          botResponse = pythonResponse.data.answer;
        }
      } catch (error) {
        console.error("❌ Error calling Python API:", error.response ? error.response.data : error.message);
        // The default error message will be used
      }
      
      console.log(`🤖 Sending response to ${from}: "${botResponse}"`);

      // 📤 SEND THE PYTHON SERVER'S RESPONSE BACK TO THE USER VIA WHATSAPP
      await axios.post(
        `https://graph.facebook.com/v19.0/${PHONE_NUMBER_ID}/messages`,
        {
          messaging_product: "whatsapp",
          to: from,
          text: { body: botResponse },
        },
        { headers: { Authorization: `Bearer ${WHATSAPP_TOKEN}` } }
      );
    }
  } catch (error) {
    console.error("❌ An error occurred in the webhook handler:", error);
  }

  // Always send a 200 OK to WhatsApp to acknowledge receipt of the event
  res.sendStatus(200);
});

// ✅ Start the gateway server
app.listen(3000, () => {
  console.log("🚀 Node.js Gateway Server is running on port 3000");
  console.log("Waiting for messages from WhatsApp...");
});