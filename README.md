# AI Voice Chatbot: Documentation

## 1. Executive Summary

This project is a voice-enabled chatbot that integrates:

*   **Speech-to-Text (STT):** Hugging Face’s Whisper
*   **Text Generation (LLM):** OpenAI GPT-4o via a custom endpoint
*   **Text-to-Speech (TTS):** gTTS

**Live Demo:** Try it on [Hugging Face Spaces](https://huggingface.co/spaces/Ashgen12/ai-voice-bot).

### Key Features

*   Real-time microphone input → AI voice response.
*   Handles API failures gracefully (e.g., missing keys, rate limits).
*   Clean Gradio UI with example prompts.

## 2. Setup & Deployment

### A. Local Setup

1.  **Install dependencies:**

    ```bash
    pip install gradio openai gtts pydub huggingface_hub
    ```

2.  **Configure API keys:**

    *   Replace `HF_API_KEY` (Hugging Face) in the script.
    *   Ensure `chat_client` has a valid API key (custom endpoint).

3.  **Run:**

    ```bash
    python3 voice_chatbot.py
    ```

    → Launches at `http://127.0.0.1:7860`.

### B. Hugging Face Spaces Deployment

*   **Pre-configured:** No setup needed.
*   **Access:** [https://huggingface.co/spaces/Ashgen12/ai-voice-bot](https://huggingface.co/spaces/Ashgen12/ai-voice-bot).

## 3. Technical Design

### A. Workflow

*   **Input:** User speech (mic/file) → Whisper transcription.
*   **Processing:** GPT-4o generates response.
*   **Output:** gTTS converts text to speech.

### B. Error Handling

| Scenario           | Response                                     |
| ------------------ | -------------------------------------------- |
| Missing API Keys   | Silent audio or TTS error message.          |
| Whisper Failure    | Returns TTS: "Transcription failed."        |
| ChatGPT Error      | Falls back to error audio clip.              |

### C. UI/UX

*   **Gradio Interface:**
    *   Mic input + autoplay response.
    *   Example questions (e.g., "What’s your #1 superpower?").

## 4. Limitations & Future Work

| Issue                 | Improvement                                  |
| --------------------- | -------------------------------------------- |
| API Dependencies      | Self-host Whisper/LLMs (e.g., Llama 3).        |
| gTTS Voice Quality    | Upgrade to ElevenLabs.                       |
| Latency               | Parallelize API calls where possible.        |

## 5. Conclusion

This project demonstrates a functional voice chatbot with a modular design, allowing easy swaps for different ASR/LLM/TTS components. While currently reliant on external APIs, it serves as a strong foundation for more scalable or self-hosted implementations. Future work should focus on reducing latency, improving voice quality, and exploring offline alternatives.

## Appendix

*   **Code:** [https://github.com/Ashgen12/ai-voive-bot/].
*   **Libraries:** `gradio`, `openai`, `gtts`, `huggingface_hub`.
