
# 🗣️ AI Voice Chatbot: Technical Analysis & Documentation 🤖

**Interact with an AI like never before!** Speak directly to it and receive real-time, synthesized voice responses.  This document outlines the design, implementation, and deployment details of our voice-enabled chatbot.

**🚀 Try the Live Demo:** [AI-Voice-Bot](https://huggingface.co/spaces/Ashgen12/ai-voice-bot) 🌟

## 1. Executive Summary

This project is a sophisticated voice-enabled chatbot, seamlessly integrating three core technologies:

*   **Speech-to-Text (STT):** Powered by Hugging Face's Whisper model, converting spoken words into text.
*   **Text Generation (LLM):** Driven by OpenAI GPT-4o (accessed via a custom API endpoint), generating intelligent and engaging responses.
*   **Text-to-Speech (TTS):** Utilizing gTTS to transform the AI's text responses into natural-sounding synthesized speech.

The result is a fluid and intuitive conversational AI experience, allowing users to engage in natural dialogues simply by speaking.

### ✨ Key Features

*   🎤 **Real-time Voice Interaction:**  Speak directly to the AI using your microphone.
*   🧠 **Whisper Speech Recognition:** Employs Hugging Face’s robust Whisper model (`openai/whisper-medium`) for accurate transcription.
*   💬 **GPT-4o Integration:** Connects to a custom OpenAI-compatible API endpoint (`https://beta.sree.shop/v1`) for generating intelligent responses.
*   🗣️ **gTTS Text-to-Speech:**  Converts AI responses into clear and synthesized speech using gTTS.
*   🛡️ **Robust Error Handling:** Implements comprehensive exception management for API failures, ensuring a stable user experience.
*   🎨 **User-Friendly Gradio UI:** Features an intuitive web interface with microphone input and audio playback.

### 💡 Potential Use Cases

*   **Personal AI Assistant:**  A voice-based Q&A system for quick access to information.
*   **Interview Simulation:**  Practice your interview skills with an AI that answers questions about itself.
*   **Accessibility Tool:**  Enables voice interaction for users with limited typing abilities.

## 2. Setup & Deployment Instructions

### ⚙️ Prerequisites

*   Python 3.8+

### 📦 Required Libraries

```bash
pip install gradio openai gtts pydub huggingface_hub
```

### 🔑 API Keys

*   Hugging Face API Key (for Whisper transcription)
*   OpenAI-Compatible API Key (for chat completions)

### 🛠️ Configuration

1.  **Replace API Keys:**

    *   Set `HF_API_KEY` (Hugging Face) in the script.
    *   Ensure `chat_client` has a valid API key (currently using the custom endpoint).

2.  **Optional Model Changes:**

    *   Modify `HF_WHISPER_MODEL` if a different Whisper variant is desired.
    *   Adjust `chat_client.base_url` if using a different OpenAI-compatible API.

### 🚀 Running the Application

1.  **Execute the script:**

    ```bash
    python3 voice_chatbot.py
    ```

2.  The Gradio interface will launch locally (typically at `http://127.0.0.1:7860`).

3.  Ensure microphone access is enabled for real-time input.

## 3. Technical Approach & Design Decisions

### A. System Architecture

#### 🎤 Input Handling

*   Accepts microphone audio input through Gradio's `gr.Audio` component.
*   Supports file uploads (though primarily designed for real-time speech).

#### 🗣️ Speech-to-Text (Whisper via Hugging Face)

*   Leverages Hugging Face's `InferenceClient` for Whisper transcription.
*   **Why Hugging Face?**
    *   Provides a free tier (unlike OpenAI's Whisper API).
    *   Offers self-hostable alternatives (e.g., `whisper.cpp`).

#### 💬 Text Generation (GPT-4o via Custom Endpoint)

*   Connects to a custom OpenAI-compatible API (`beta.sree.shop/v1`).
*   **Prompt Engineering:**
    *   **System prompt:**  Primes GPT to respond introspectively (e.g., "What's your #1 superpower?").
    *   **Temperature (0.7):** Balances creativity and coherence.

#### 🗣️ Text-to-Speech (gTTS)

*   Converts ChatGPT's response into synthesized speech.
*   **Why gTTS?**
    *   It's free and doesn't require an API key.
    *   It's lightweight compared to alternatives like ElevenLabs.

#### 📢 Output Delivery

*   Returns audio in MP3 format. (Conversion to WAV is possible using `pydub` if needed).

### B. Error Handling & Edge Cases

| Scenario                        | Handling Strategy                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| Missing API Keys                | Warns during startup; returns silent audio or an error TTS message.              |
| Whisper Transcription Failure   | Catches `HfHubHTTPError`, returns a TTS error message.                                |
| ChatGPT API Failure             | Handles `openai.AuthenticationError`, `RateLimitError`, etc.; fails gracefully.   |
| Empty User Input                | Returns a short silent audio clip to prevent crashes.                               |
| TTS Generation Failure          | Falls back to returning `None` (Gradio handles missing output gracefully).          |

### C. UI/UX Design (Gradio)

*   **Visual Styling:**
    *   Clean and modern interface with gradient headers and soft colors.
    *   Example questions guide users and spark conversation.
*   **Real-Time Interaction:**
    *   Audio autoplay ensures a seamless and engaging conversational flow.
*   **Accessibility:**
    *   Microphone input lowers barriers for non-technical users, promoting inclusivity.


### 4. Limitations & Future Work

| Issue                 | Improvement                                  |
| --------------------- | -------------------------------------------- |
| API Dependencies      | Self-host Whisper/LLMs (e.g., Llama 3).        |
| gTTS Voice Quality    | Upgrade to ElevenLabs.                       |
| Latency               | Parallelize API calls where possible.        |

### 5. Conclusion

This project demonstrates a functional voice chatbot with a modular design, allowing easy swaps for different ASR/LLM/TTS components. While currently reliant on external APIs, it serves as a strong foundation for more scalable or self-hosted implementations. Future work should focus on reducing latency, improving voice quality, and exploring offline alternatives.

### Appendix

*   **Code:** [https://github.com/Ashgen12/ai-voive-bot/].
*   **Libraries:** `gradio`, `openai`, `gtts`, `huggingface_hub`.
