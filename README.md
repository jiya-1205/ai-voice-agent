# ai-voice-agent
# 🎙️ AI Voice Agent System (AWS-Based)

A real-time, multilingual AI voice agent designed to simulate
customer-facing phone conversations using speech, LLMs, and memory.

This repository contains **two AWS-based variants** of the same core system,
each optimized for a different real-world use case.

---

## ✨ Key Capabilities
- 🎧 Real-time Speech-to-Text
- 🗣️ Natural Text-to-Speech responses
- 🧠 LLM-powered conversation handling
- 🌍 Multilingual voice support
- 🔁 Turn-based call flow
- 📝 Automatic call summaries & records
- ☎️ Manual and AI-driven call termination

---

## 🧩 System Variants

### 1️⃣ AWS Product Feedback Voice Agent
**Location:** `aws_product_feedback/`

**Purpose:**  
Designed to follow a **structured product feedback workflow**, similar to
enterprise customer-feedback calls.

**Highlights:**
- Fixed conversational phases (greeting → usage → feedback → closure)
- Multilingual support (Indian languages)
- AWS Bedrock (LLaMA 3) for response generation
- Automatic post-call summaries
- Clean session-based call handling

**Use case:**  
Enterprise product feedback, surveys, controlled customer outreach.

---

### 2️⃣ AWS General Customer Support Voice Agent
**Location:** `aws_general_support/`

**Purpose:**  
A **general-purpose AI support agent** that adapts to user queries and
learns from each interaction.

**Highlights:**
- Flexible, intent-driven conversation flow
- Local knowledge base + LLM fallback
- Learns from every conversation
- Saves transcripts and summaries automatically
- Designed for extensibility and experimentation

**Use case:**  
Customer support, helpdesks, conversational AI demos.

---

## 🏗️ High-Level Architecture

