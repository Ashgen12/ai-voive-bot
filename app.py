#!/usr/bin/env python3
import os
import tempfile
import gradio as gr
import openai 
from openai import OpenAI 
from gtts import gTTS
from pydub import AudioSegment
from huggingface_hub import InferenceClient 
from huggingface_hub.utils import HfHubHTTPError 

# --- Configuration ---
HF_API_KEY = ''
# ---

# Initialize the OpenAI client FOR CHAT COMPLETIONS
chat_client = OpenAI(
    api_key="",
    base_url="https://beta.sree.shop/v1" # This is for CHAT
)

# Initialize the Hugging Face client FOR WHISPER TRANSCRIPTION
HF_WHISPER_MODEL = "openai/whisper-medium" # You can change this model if needed (e.g., "openai/whisper-medium")
hf_inference_client = None
try:
    if not HF_API_KEY or HF_API_KEY == 'hf_YOUR_ACTUAL_HUGGINGFACE_TOKEN':
        print("WARNING: Hugging Face API Key (HF_API_KEY) is missing or still the placeholder.")
        print(" -> Transcription will likely fail. Please replace the placeholder in the script.")
    

    # Initialize using the specified pattern, explicitly passing the API key
    hf_inference_client = InferenceClient(
        # provider="hf-inference", # Optional: provider can often be inferred
        api_key=HF_API_KEY # Pass the key directly
    )
    print(f"Hugging Face Inference Client initialized (explicit key) for model: {HF_WHISPER_MODEL}")

except ValueError as e:
    # Specific check for token missing or invalid format during init
    print(f"Error initializing HuggingFace Inference Client: {e}")
    print("Please ensure 'hf_YOUR_ACTUAL_HUGGINGFACE_TOKEN' in the script is replaced with a valid Hugging Face token.")
except Exception as e:
    print(f"Unexpected error initializing HuggingFace Inference Client: {e}")
    print("Transcription via Hugging Face will likely fail.")
    print("Ensure 'huggingface_hub' is installed.")


# --- Main Function ---

def voice_chatbot(audio_file):
    """
    This function performs the following steps:
    1. Transcribes the incoming audio using Hugging Face's Inference API (Whisper model).
    2. Sends the transcribed text as a prompt to the ChatGPT API (via your custom endpoint).
    3. Converts the generated text response to speech using gTTS.
    4. Converts the gTTS-generated mp3 file to WAV format.
    5. Returns the path to the WAV file containing the answer audio.
    """
    if audio_file is None:
        print("No audio file received.")
        return None

    # Check if Hugging Face client is available
    if hf_inference_client is None:
        print("Hugging Face client not initialized. Cannot transcribe.")
        return None # Or return TTS error audio

    mp3_path = None
    wav_path = None

    try:
        # Step 1: Transcribe audio using Hugging Face Inference API
        print(f"Transcribing audio using Hugging Face model: {HF_WHISPER_MODEL}...")
        transcription_response = hf_inference_client.automatic_speech_recognition(
            audio=audio_file,
            model=HF_WHISPER_MODEL # Specify model here if not default for client
        )
        transcript = transcription_response.get('text', '').strip()
        print(f"HF Transcription: {transcript}")

        if not transcript:
            print("Empty transcript received from Hugging Face.")
            silent_audio = AudioSegment.silent(duration=100) # 100ms silent wav
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                 silent_audio.export(tmp_wav.name, format="wav")
                 return tmp_wav.name
            # return None

        # Step 2: Call ChatGPT API (using your separate OpenAI client and endpoint)
        print(f"Sending prompt to ChatGPT endpoint: {chat_client.base_url}")
        messages = [
            {"role": "system", "content": (
                "You are ChatGPT, a large language model who answers questions in a friendly and "
                "detailed manner. Pretend the following audio query is directed to you. "
                "Respond as if you were answering these questions about yourself:"
                "\n • What should we know about your life story in a few sentences?"
                "\n • What's your #1 superpower?"
                "\n • What are the top 3 areas you'd like to grow in?"
                "\n • What misconception do your coworkers have about you?"
                "\n • How do you push your boundaries and limits?"
            )},
            {"role": "user", "content": transcript}
        ]
        chat_response = chat_client.chat.completions.create(
            model="Provider-5/gpt-4o",
            max_tokens=2048,
            messages=messages,
            temperature=0.7
        )
        answer_text = chat_response.choices[0].message.content.strip()
        print(f"ChatGPT Answer: {answer_text}")

        # Step 3: Convert text answer to speech using gTTS
        print("Generating speech using gTTS...")
        tts = gTTS(text=answer_text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            tts.save(tmp_mp3.name)
            mp3_path = tmp_mp3.name

        print(f"Returning MP3 file: {mp3_path}")
        return mp3_path

    # --- Updated Exception Handling ---
    except HfHubHTTPError as e:
        error_message = f"Hugging Face API Error during transcription: {e}"
        print(error_message)
        status_code = getattr(getattr(e, 'response', None), 'status_code', None)
        if status_code == 401:
            error_message = "Authentication failed with Hugging Face. Check the HF_API_KEY in the script."
            print(error_message)
        elif status_code == 429:
             error_message = "Rate limit likely exceeded on Hugging Face free tier."
             print(error_message)
        else:
             error_message = f"Problem connecting to Hugging Face model {HF_WHISPER_MODEL}. Status code: {status_code}."
             print(error_message)
        try:
            error_text = f"Sorry, transcription failed: {e}" # Simplified error text
            tts_error = gTTS(text=error_text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3_err:
                tts_error.save(tmp_mp3_err.name)
            # No conversion needed, return the error MP3
            return tmp_mp3_err.name
        except Exception as tts_e:
            print(f"Could not generate TTS for error message: {tts_e}")
            return None

    except openai.AuthenticationError:
        error_message = f"OpenAI API Error (for Chat @ {chat_client.base_url}): Authentication failed."
        print(error_message)
        return None # Or return TTS error audio
    except openai.RateLimitError:
        error_message = f"OpenAI API Error (for Chat @ {chat_client.base_url}): Rate limit exceeded."
        print(error_message)
        return None # Or return TTS error audio
    except openai.NotFoundError:
         error_message = f"OpenAI API Error (for Chat @ {chat_client.base_url}): Chat model/endpoint not found."
         print(error_message)
         return None # Or return TTS error audio
    except openai.APIConnectionError as e:
        error_message = f"OpenAI API Error (for Chat @ {chat_client.base_url}): Connection error: {e}"
        print(error_message)
        return None # Or return TTS error audio
    except Exception as e:
        print(f"An unexpected error occurred in voice_chatbot: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        # Try to return a general error message as audio
        try:
            tts_error = gTTS(text="Sorry, an unexpected error occurred.", lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3_err:
                tts_error.save(tmp_mp3_err.name)
            return tmp_mp3_err.name
        except Exception as tts_e:
            print(f"Could not generate TTS for general error message: {tts_e}")
            return None


# --- Gradio UI (Exactly as before) ---
custom_css = """
.gradio-container { font-family: 'Helvetica Neue', Arial, sans-serif; }
.header { text-align: center; margin-bottom: 20px; }
.header h1 { font-size: 2.5rem; margin-bottom: 10px; background: linear-gradient(90deg, #4f46e5, #06b6d4); -webkit-background-clip: text; background-clip: text; color: transparent; }
.header p { font-size: 1.1rem; color: #4b5563; }
.example-container { background: #f9fafb; padding: 15px; border-radius: 10px; margin: 15px 0; }
.example-container h3 { margin-top: 0; color: #4f46e5; }
.example-list { padding-left: 20px; }
.example-list li { margin-bottom: 8px; }
.audio-input { background: #f3f4f6; padding: 20px; border-radius: 10px; }
.audio-output { background: #ecfdf5; padding: 20px; border-radius: 10px; }
footer { text-align: center; margin-top: 20px; color: #6b7280; font-size: 0.9rem; }
"""
header_html = """
<div class="header"><h1>🎙️ AI Voice Bot</h1><p>Speak naturally and get responses in ChatGPT's voice</p></div>
"""
examples_html = """
<div class="example-container"><h3>Try asking:</h3><ul class="example-list">
<li>What should we know about your life story in a few sentences?</li><li>What's your #1 superpower?</li>
<li>What are the top 3 areas you'd like to grow in?</li><li>What misconception do your coworkers have about you?</li>
<li>How do you push your boundaries and limits?</li></ul></div>
"""
footer_html = """
<div class="footer"><p>Powered by Ashgen12</p></div>
"""

# Build the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(header_html)
    gr.HTML(examples_html)
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your question", show_label=True, elem_classes="audio-input")
        with gr.Column(scale=1):
            audio_output = gr.Audio(type="filepath", label="Response", elem_classes="audio-output", autoplay=True)
    gr.HTML(footer_html)
    audio_input.change(
        voice_chatbot,
        inputs=audio_input,
        outputs=audio_output,
        api_name=False
    )

# --- Launch ---
if __name__ == "__main__":
    print("Starting application...")

    # Check Chat client API key placeholder
    if not chat_client.api_key or chat_client.api_key == "ddc-beta-v7bjela50v-xx":
        print("*********************************************************************")
        print("WARNING: OpenAI API key for CHAT might be placeholder or missing.")
        print(f"Using base_url for CHAT: {chat_client.base_url}")
        print("*********************************************************************")

    # Check Hugging Face client initialization and API Key placeholder
    if hf_inference_client is None:
         print("*********************************************************************")
         print("CRITICAL WARNING: Hugging Face Inference Client failed to initialize.")
         print(" -> Whisper transcription via Hugging Face WILL NOT WORK.")
         print(" -> Check errors printed during initialization above.")
         print(" -> Ensure 'huggingface_hub' is installed (`pip install huggingface_hub`).")
         if not HF_API_KEY or HF_API_KEY == 'hf_YOUR_ACTUAL_HUGGINGFACE_TOKEN':
              print(" -> Reason: HF_API_KEY is likely missing or still the placeholder value.")
              print(" -> Please replace 'hf_YOUR_ACTUAL_HUGGINGFACE_TOKEN' in the script.")
         print("*********************************************************************")
    elif HF_API_KEY == 'hf_YOUR_ACTUAL_HUGGINGFACE_TOKEN':
         # This case covers if initialization *succeeded* somehow but the placeholder is still there
         print("*********************************************************************")
         print("WARNING: Hugging Face API Key (HF_API_KEY) appears to be the placeholder.")
         print(" -> Please replace 'hf_YOUR_ACTUAL_HUGGINGFACE_TOKEN' in the script.")
         print(" -> Transcription will likely fail with an authentication error later.")
         print("*********************************************************************")

    print("\nLaunching Gradio Interface...")
    demo.launch()