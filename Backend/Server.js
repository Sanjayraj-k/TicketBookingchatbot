const express = require('express');
const mongoose = require('mongoose');
const Razorpay = require('razorpay');
const path = require('path');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
const QRCode = require('qrcode');

const app = express();
const port = 4010;

app.use(express.static(__dirname));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// MongoDB connection
mongoose.connect("mongodb://localhost:27017/Login", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
}).then(() => {
    console.log("MongoDB connection established successfully");
}).catch((err) => {
    console.error("MongoDB connection error:", err);
});

// MongoDB schema
const UserSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true },
    ticketCount: { type: Number, required: true },
    ticketAmount: { type: Number, required: true },
    eventDate: { type: Date, required: true },
    paymentId: { type: String, required: true },
    paymentStatus: { type: String, required: true },
    qrCode: { type: String, required: true }, // Added for storing QR code
    timestamp: { type: Date, default: Date.now },
});


const User = mongoose.model("Payment", UserSchema);

// Razorpay instance
const razorpay = new Razorpay({
    key_id: 'rzp_test_1Ss2OE5DsbSMr0', 
    key_secret: 'PSwn48wSWKAD0HwJptCOXoUt', 
});

// Nodemailer setup
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'shimalvip@gmail.com',
        pass: 'uhhu yiua yjrn ifjn',
    },
});

// Serve HTML page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'design.html'));
});

// Create Razorpay order
app.post('/create-order', async (req, res) => {
    const { amount } = req.body;
    const options = {
        amount: amount * 100,
        currency: "INR",
        receipt: `receipt_${Date.now()}`,
    };

    try {
        const order = await razorpay.orders.create(options);
        res.status(200).json({ success: true, order });
    } catch (error) {
        console.error("Error creating Razorpay order:", error);
        res.status(500).json({ success: false, message: "Unable to create order" });
    }
});

// Handle payment success
app.post('/payment-success', async (req, res) => {
    const { name, email, ticket, tickets, date, paymentId, paymentStatus } = req.body;

    try {
        const ticketCount = parseInt(tickets);
        const ticketAmount = parseInt(ticket);

        QRCode.toDataURL(paymentId, async (err, qrCodeDataUrl) => {
            if (err) {
                console.error("Error generating QR code:", err);
                return res.status(500).send("Failed to generate QR code");
            }

            // Save user details with QR code to the database
            const user = new User({
                name,
                email,
                ticketCount,
                ticketAmount,
                eventDate: new Date(date),
                paymentId,
                paymentStatus,
                qrCode: qrCodeDataUrl, // Save the QR code data
            });

            await user.save();

            // Send email with QR code as an attachment
            const qrCodeBuffer = Buffer.from(qrCodeDataUrl.split(",")[1], "base64"); // Convert QR code to a buffer

            const mailOptions = {
                from: 'shimalvip@gmail.com',
                to: email,
                subject: 'Payment Receipt - Museum Tickets',
                html: `
                    <h2>Payment Successful</h2>
                    <p>Dear ${name},</p>
                    <p>Thank you for booking your tickets with us. Here are your payment details:</p>
                    <ul>
                        <li><strong>Payment ID:</strong> ${paymentId}</li>
                        <li><strong>Tickets Booked:</strong> ${ticketCount}</li>
                        <li><strong>Total Amount:</strong> ₹${ticketAmount}</li>
                        <li><strong>Event Date:</strong> ${new Date(date).toLocaleDateString()}</li>
                        <li><strong>Payment Status:</strong> ${paymentStatus}</li>
                    </ul>
                    <p>We look forward to seeing you at the event!</p>
                    <p>The QR code for your payment is attached to this email.</p>
                    <p>Best Regards,<br>Museum Tickets Team</p>
                `,
                attachments: [
                    {
                        filename: 'payment-qr-code.png',
                        content: qrCodeBuffer,
                        contentType: 'image/png',
                    },
                ],
            };

            transporter.sendMail(mailOptions, (err, info) => {
                if (err) {
                    console.error("Error sending email:", err);
                } else {
                    console.log("Email sent successfully:", info.response);
                }
            });

            res.status(200).send("Payment successful, details saved, and receipt with QR code sent!");
        });
    } catch (error) {
        console.error("Error processing payment or sending email:", error);
        res.status(500).send("Failed to save payment details or send email");
    }
});


// Validate ticket
app.post('/validate', async (req, res) => {
    const { name, email, ticket, date } = req.body;

    try {
        const user = await User.findOne({
            name,
            email,
            ticketCount: ticket,
            eventDate: new Date(date),
        });

        if (user) {
            res.send("Ticket validated successfully!");
        } else {
            res.send("Ticket not found. Please check your details.");
        }
    } catch (error) {
        console.error("Validation error:", error);
        res.status(500).send("An error occurred while validating the ticket.");
    }
});
app.get('/user-details/:paymentId', async (req, res) => {
    const { paymentId } = req.params;

    try {
        const user = await User.findOne({ paymentId });

        if (user) {
            res.json({
                name: user.name,
                email: user.email,
                ticketCount: user.ticketCount,
                qrCode: user.qrCode, // Send QR code data if needed
            });
        } else {
            res.status(404).send("User not found");
        }
    } catch (error) {
        console.error("Error fetching user details:", error);
        res.status(500).send("An error occurred");
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server started on http://localhost:${port}`);
});
