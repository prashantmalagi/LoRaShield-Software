# LoRaShield-Software

AI-powered Secure Morse Code Communication System using LoRa Technology

---

## Overview

LoRaShield is a secure communication platform that combines Artificial Intelligence, AES Encryption, and LoRa technology for transmitting encrypted Morse code messages over long distances.

The software provides an intuitive desktop GUI for encoding, decoding, encrypting, and managing secure communications. It is being developed as part of the NAIN (New Age Innovation Network) project.

---

## Current Status

### Software
- AI-powered Morse Decoder
- Morse Code Encoder
- AES-256 Encryption & Decryption
- Modern Desktop GUI
- Dashboard
- Activity Logs
- Message Statistics
- Simulation Mode
- TensorFlow Model Integration

### Hardware (In Progress)
- ESP32 Integration
- SX1278 LoRa Module
- OLED Display
- Real-time LoRa Communication

---

## Features

- Morse Code Encoding
- AI-based Morse Code Decoding
- AES Secure Encryption
- Dashboard with System Status
- Message Logs
- Encryption Statistics
- LoRa Communication (Simulation)
- Modern Dark UI
- Modular Architecture

---

## Project Structure

```
LoRaShield-Software
│
├── assets/
├── data/
├── models/
│   ├── morse_decoder.h5
│   ├── tokenizer.pkl
│   ├── encoder.pkl
│
├── ui/
├── utils/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/prashantmalagi/LoRaShield-Software.git
```

Move into the project

```bash
cd LoRaShield-Software
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## Technology Stack

### Programming Language

- Python

### Machine Learning

- TensorFlow
- Keras
- NumPy
- Scikit-learn

### GUI

- Tkinter
- ttkbootstrap

### Security

- AES Encryption (PyCryptodome)

### Communication

- LoRa SX1278 (In Progress)
- ESP32 (In Progress)

---

## Workflow

```
Text
   ↓
Morse Encoding
   ↓
AES Encryption
   ↓
LoRa Transmission
   ↓
AES Decryption
   ↓
AI Morse Decoder
   ↓
Recovered Text
```

---

## Current Development Progress

| Module | Status |
|---------|--------|
| GUI | Completed |
| Dashboard | Completed |
| Morse Encoder | Completed |
| AI Decoder | Integrated |
| AES Encryption | Completed |
| Activity Logs | Completed |
| Simulation Mode | Completed |
| ESP32 Integration | In Progress |
| LoRa Communication | In Progress |
| OLED Display | In Progress |

---

## Future Improvements

- Real-time ESP32 Communication
- Live LoRa Packet Monitoring
- Signal Strength Analysis
- AI Model Optimization
- Noise-resistant Morse Recognition
- Database Integration
- User Authentication

---

## Team

NAIN Project Team

Software Development
- Prashanth Malagi

Hardware Development
- Team Members

---

## License

This project is developed for academic and research purposes under the NAIN Innovation Project.
