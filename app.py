from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse
from flask_cors import CORS
import requests
import os
import traceback
import google.generativeai as genai
import logging
from dotenv import load_dotenv
import urllib.parse
import html
import json
import base64
from google.cloud import texttospeech
from twilio.rest import Client

# ✅ Load API keys securely
load_dotenv(dotenv_path="tests/apis.env")

# ✅ Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(_name_)

app = Flask(_name_)
CORS(app)

# ✅ Securely load API keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")  # Still kept for speech-to-text
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# ✅ API URLs - Only keeping Sarvam for speech-to-text
SARVAM_SPEECH_TO_TEXT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

# ✅ Configure Google Gemini AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('models/gemini-1.5-pro-002')
else:
    model = None
    logger.warning("Google API Key is missing!")

# ✅ Initialize Google Cloud TTS client
try:
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    logger.error(f"Failed to initialize Google Text-to-Speech client: {e}")
    tts_client = None

def get_gemini_response(user_message, language):
    """Generates a response using Gemini AI with financial advice emphasis."""
    if not GOOGLE_API_KEY or not model:
        return ["Sorry, the service is unavailable."]
    try:
        logger.debug(f"Processing user message: {user_message}")

        # ✅ Handle Greetings
        if any(greet in user_message.lower() for greet in ["hello", "hi", "hey", "नमस्ते", "हैलो"]):
            if language == "en":
                return ["Hello! How can I assist you today?"]
            elif language == "hi":
                return ["नमस्ते! मैं आपकी कैसे सहायता कर सकता हूँ?"]
            else:
                # Get greeting in the detected language using translation
                greeting = translate_text("Hello! How can I assist you today?", source_lang="en", target_lang=language)
                return [greeting]

        # ✅ Financial Expert Response
        prompt = (
            f"You are a financial expert providing concise guidance on loans, banking, and financial management. "
            f"Reply in {language} with short, specific points. If user asks for financial advice, list key points. "
            f"Keep answers brief and token-efficient.\n\nUser Query: {user_message}"
            f"Your responses should be clear, concise, and tailored to the user's query. "
            f"Always prioritize accuracy and relevance. "
            f"Provide a clear response in {language}."
        )

        response = model.generate_content(prompt)
        return [response.text.strip()] if response else ["No response generated."]
    
    except Exception as e:
        logger.error(f"Gemini response error: {e}")
        return ["Error processing your request."]

# ✅ Function: Convert Speech to Text & Translate (Kept Sarvam for this function)
def speech_to_text_translate(audio_url, target_lang="auto", api_key=SARVAM_API_KEY):
    """Downloads audio from a given URL, then sends it to the Sarvam API for speech-to-text translation."""
    try:
        audio_response = requests.get(audio_url, stream=True)
        if audio_response.status_code != 200:
            return "Error downloading audio file."

        audio_file_path = "temp_audio.wav"
        with open(audio_file_path, "wb") as audio_file:
            for chunk in audio_response.iter_content(chunk_size=1024):
                audio_file.write(chunk)

        with open(audio_file_path, "rb") as file:
            files = {"file": (audio_file_path, file, "audio/wav")}
            # Don't translate, just transcribe in original language
            payload = {"with_diarization": "false", "model": "saaras:v2", "target_language_code": target_lang}
            headers = {"api-subscription-key": api_key}

            response = requests.post(SARVAM_SPEECH_TO_TEXT_TRANSLATE_URL, files=files, data=payload, headers=headers)

        os.remove(audio_file_path)

        if response.status_code == 200:
            return response.json().get("transcript", "Error extracting transcript.")
        else:
            return f"Error in speech translation: {response.text}"
    except Exception as e:
        return f"Speech translation service error: {str(e)}"

def translate_text(text, source_lang="auto", target_lang="hi"):
    """Translates text using Google Translate API without authentication."""
    try:
        # Clean up language codes from formats like 'hi-IN' to 'hi'
        if source_lang != "auto" and "-" in source_lang:
            source_lang = source_lang.split("-")[0]
        
        if "-" in target_lang:
            target_lang = target_lang.split("-")[0]
        
        # Encode the text for URL
        encoded_text = urllib.parse.quote(text)
        
        # Use the free Google Translate API
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={encoded_text}"
        
        logger.debug(f"Translation request: {source_lang} → {target_lang}")
        
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            
            # Extract the translated text
            translated_text = ''
            for sentence in result[0]:
                if sentence[0]:
                    translated_text += sentence[0]
            
            # Decode HTML entities
            translated_text = html.unescape(translated_text)
            
            logger.debug(f"Translation successful: {translated_text[:50]}...")
            return translated_text
        else:
            logger.error(f"Translation error: {response.status_code}")
            return f"Translation error ({response.status_code}). Original text: {text[:30]}..."
            
    except Exception as e:
        logger.error(f"Translation service error: {str(e)}")
        return f"Translation error: {str(e)}. Original text: {text[:30]}..."

def detect_language(text):
    """Detects the language of the given text using Google Translate."""
    try:
        # Use the first few words for detection to save bandwidth
        sample = text[:100] if len(text) > 100 else text
        encoded_sample = urllib.parse.quote(sample)
        
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_sample}"
        response = requests.get(url)
        
        if response.status_code == 200:
            # The detected language is in the third item of the response
            detected_lang = response.json()[2]
            logger.debug(f"Detected language: {detected_lang}")
            return detected_lang
        else:
            logger.error(f"Language detection error: {response.status_code}")
            return "en"  # Default fallback
            
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "en"  # Default fallback

def detect_language_robust(text):
    """More robust language detection with character set analysis as backup."""
    # First try the API-based detection
    api_detected = detect_language(text)
    logger.debug(f"API detected language: {api_detected}")
    
    # Character set analysis as backup/validation
    # Hindi character detection
    hindi_chars = "अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
    hindi_char_count = sum(1 for char in text if char in hindi_chars)
    
    # If significant Hindi characters are present but language was detected as English
    # This can happen with short texts or when speech recognition isn't perfect
    if api_detected == "en" and hindi_char_count > 3:
        logger.debug(f"Overriding language detection from {api_detected} to hi based on {hindi_char_count} Hindi characters")
        return "hi"
    
    return api_detected

# ✅ NEW FUNCTION: Text-to-Speech using Google Cloud TTS
def text_to_speech(text, language_code="en-US"):
    """Converts text to speech using Google Cloud Text-to-Speech."""
    try:
        if not tts_client:
            logger.error("Google TTS client not initialized")
            return None
            
        # Map language codes to appropriate TTS voice language codes
        voice_language_map = {
            "en": "en-US",
            "hi": "hi-IN",
            "fr": "fr-FR",
            "es": "es-ES",
            "de": "de-DE",
            "zh": "cmn-CN",
            "ja": "ja-JP",
            "ru": "ru-RU",
            "ar": "ar-XA",
            "bn": "bn-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "kn": "kn-IN",
            "ml": "ml-IN",
        }
        
        # Get appropriate voice language code
        if language_code in voice_language_map:
            voice_language = voice_language_map[language_code]
        elif "-" in language_code:
            base_lang = language_code.split("-")[0]
            voice_language = voice_language_map.get(base_lang, "en-US")
        else:
            voice_language = "en-US"
            
        logger.debug(f"Converting text to speech in {voice_language}")
        
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_language,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Perform the text-to-speech request
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Save the audio to a temp file
        temp_audio_path = f"temp_tts_{language_code}.mp3"
        with open(temp_audio_path, "wb") as out:
            out.write(response.audio_content)
            
        logger.debug(f"TTS audio saved to {temp_audio_path}")
        return temp_audio_path
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ✅ Alternative TTS function using gTTS if Google Cloud TTS is not available
def text_to_speech_gtts(text, language_code="en"):
    """Fallback method to convert text to speech using gTTS."""
    try:
        from gtts import gTTS
        
        # Clean up language code
        if "-" in language_code:
            language_code = language_code.split("-")[0]
            
        # Create gTTS object
        tts = gTTS(text=text, lang=language_code, slow=False)
        
        # Save to file
        temp_audio_path = f"temp_tts_{language_code}.mp3"
        tts.save(temp_audio_path)
        
        logger.debug(f"gTTS audio saved to {temp_audio_path}")
        return temp_audio_path
        
    except Exception as e:
        logger.error(f"gTTS error: {e}")
        return None

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    try:
        incoming_msg = request.values.get("Body", "").strip()
        sender = request.values.get("From", "")
        media_url = request.values.get("MediaUrl0", None)  # Check for audio file
        
        # Check if the user wants a voice response
         # Create Twilio client
   
# Check if the user wants a voice response
        voice_mode = "voice" in incoming_msg.lower() or media_url is not None
        logger.info(f"Received message from {sender}: {incoming_msg}")

        response_msg = "I'm sorry, I didn't understand your request."  # ✅ Default response
        detected_lang = "en"  # ✅ Default to English
        should_send_voice = False

        # ✅ Handle direct translation requests
        if incoming_msg.lower().startswith("translate to "):
            try:
                target_language, text_to_translate = incoming_msg[12:].split(":", 1)
                target_language = target_language.strip()
                text_to_translate = text_to_translate.strip()

                # Detect source language
                detected_lang = detect_language(text_to_translate)
                logger.debug(f"Source text language detected as: {detected_lang}")
                
                # Skip translation if source and target languages are the same
                if detected_lang == target_language:
                    response_msg = f"No translation needed: {text_to_translate}"
                else:
                    translated_text = translate_text(text_to_translate, source_lang=detected_lang, target_lang=target_language)
                    response_msg = f"Translated: {translated_text}"
                    detected_lang = target_language  # Set detected language to target for TTS
                    logger.debug(f"Translation complete: {translated_text[:50]}...")
                
                # If translation was requested through voice, respond with voice
                should_send_voice = voice_mode
                
            except ValueError:
                response_msg = "Invalid translation format. Use: 'Translate to <language>: <text>'"

        # ✅ Voice Message Processing
        elif media_url:
            logger.debug("Processing voice message...")
            # Set target_lang to "auto" to maintain original language
            transcribed_text = speech_to_text_translate(media_url, "auto")
            logger.info(f"Transcribed Audio: {transcribed_text}")

            if transcribed_text and "Error" not in transcribed_text:
                # Use robust language detection
                detected_lang = detect_language_robust(transcribed_text)
                logger.debug(f"Robust language detection from audio: {detected_lang}")
                
                # Store the original transcript
                original_transcript = transcribed_text
                
                # For debugging only - log what language Gemini will respond in
                logger.debug(f"Will request Gemini response in: {detected_lang}")
                
                # If not English, translate to English for Gemini
                if detected_lang != "en":
                    english_text = translate_text(transcribed_text, source_lang=detected_lang, target_lang="en")
                    logger.debug(f"Translated transcript to English: {english_text}")
                else:
                    english_text = transcribed_text
                
                # Get response from Gemini in English
                gemini_responses = get_gemini_response(english_text, "en")
                gemini_text = "\n".join(gemini_responses)
                
                # Translate response back to detected language (only if different)
                if detected_lang == "en":
                    response_msg = gemini_text
                else:
                    logger.debug(f"Translating Gemini response to {detected_lang}")
                    translated_response = translate_text(gemini_text, source_lang="en", target_lang=detected_lang)
                    response_msg = translated_response
                
                # Since input was voice, reply with voice
                should_send_voice = True
            else:
                # Multilingual error handling
                default_error = "Sorry, I couldn't process the audio. Please try again."
                hindi_error = "क्षमा करें, मैं ऑडियो को प्रोसेस नहीं कर सका। कृपया पुनः प्रयास करें।"
                
                # Try to determine most likely user language from previous interactions or default
                likely_lang = detected_lang if detected_lang else "en"
                
                if likely_lang == "hi":
                    response_msg = hindi_error
                elif likely_lang == "en":
                    response_msg = default_error
                else:
                    # Try to translate error to detected language
                    response_msg = translate_text(default_error, source_lang="en", target_lang=likely_lang)

        # ✅ Text Message Processing with Gemini AI
        elif incoming_msg:
            # Check if the user specifically requested voice output
            if incoming_msg.lower().startswith("voice:"):
                should_send_voice = True
                incoming_msg = incoming_msg[6:].strip()
            
            # Detect user's language
            detected_lang = detect_language_robust(incoming_msg)
            logger.debug(f"Detected message language: {detected_lang}")
            
            # If not English, translate to English for Gemini
            if detected_lang != "en":
                english_query = translate_text(incoming_msg, source_lang=detected_lang, target_lang="en")
                logger.debug(f"Translated to English: {english_query}")
            else:
                english_query = incoming_msg
            
            # Get Gemini response in English
            gemini_responses = get_gemini_response(english_query, "en")
            gemini_text = "\n".join(gemini_responses)
            
            # Translate response back to original language if needed
            if detected_lang == "en":
                response_msg = gemini_text
            else:
                logger.debug(f"Translating Gemini response to {detected_lang}")
                translated_response = translate_text(gemini_text, source_lang="en", target_lang=detected_lang)
                response_msg = translated_response

        # Create TwiML response
        resp = MessagingResponse()
        
        # If voice response was requested or the input was voice
        if should_send_voice:
            try:
                # Try Google Cloud TTS first
                audio_file_path = text_to_speech(response_msg, detected_lang)
                
                # If Google Cloud TTS failed, try gTTS as fallback
                if not audio_file_path and "gtts" in globals():
                    audio_file_path = text_to_speech_gtts(response_msg, detected_lang)
                    
                if audio_file_path and os.path.exists(audio_file_path):
                    # Add text response first
                    resp.message(response_msg)
                    
                    # Create Twilio client
                    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                    with open(audio_file_path, 'rb') as audio_file:
                        twilio_media = client.messages.media.create(
                            message_sid=request.values.get("MessageSid"),
                            content_type='audio/mp3',
                            media=audio_file.read()
                        )
                    
                    # Add media to response using Twilio's media URL
                    message = resp.message("")
                    message.media(twilio_media.uri)
                    
                    logger.debug(f"Sending voice response in {detected_lang}")
                    
                    # Clean up the local file after uploading
                    os.remove(audio_file_path)
                
                if audio_file_path and os.path.exists(audio_file_path):
                    # Add text response first
                    resp.message(response_msg)
                    
                    # Add voice message if TTS was successful
                    message = resp.message("")
                    message.media(f"file://{os.path.abspath(audio_file_path)}")
                    
                    logger.debug(f"Sending voice response in {detected_lang}")
                else:
                    # Fallback to text-only if TTS failed
                    resp.message(f"{response_msg}\n\n(Voice response not available)")
                    logger.warning("Voice response requested but TTS failed")
            except Exception as tts_error:
                logger.error(f"TTS response error: {tts_error}")
                resp.message(f"{response_msg}\n\n(Voice response not available)")
        else:
            # Standard text response
            resp.message(response_msg)
        
        logger.debug(f"Replying: {response_msg[:100]}...")
        
        # Clean up temporary files
        try:
            for filename in os.listdir('.'):
                if filename.startswith('temp_tts_') and filename.endswith('.mp3'):
                    if os.path.getmtime(filename) < (time.time() - 300):  # 5 minutes old
                        os.remove(filename)
                        logger.debug(f"Cleaned up old file: {filename}")
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up temp files: {cleanup_error}")

        return str(resp)

    except Exception as e:
        logger.critical(f"Unhandled error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal Server Error"}), 500

# ✅ Route: External API for Text Translation
@app.route("/translate-text", methods=["POST"])
def translate_text_route():
    data = request.json
    text = data.get("text", "")
    source_lang = data.get("source_language", "auto")
    target_lang = data.get("target_language", "hi")
    
    # Log the translation request
    logger.debug(f"Translation API request: {source_lang} → {target_lang}, text: {text[:50]}...")
    
    translated_text = translate_text(text, source_lang=source_lang, target_lang=target_lang)
    return jsonify({"translated_text": translated_text})

# ✅ NEW ROUTE: TTS API for external use
@app.route("/text-to-speech", methods=["POST"])
def text_to_speech_route():
    try:
        data = request.json
        text = data.get("text", "")
        language = data.get("language", "en")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # Generate speech
        audio_path = text_to_speech(text, language)
        
        if not audio_path:
            # Try fallback
            audio_path = text_to_speech_gtts(text, language)
            
        if not audio_path:
            return jsonify({"error": "Failed to generate speech"}), 500
            
        # Read audio file and encode as base64
        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
            
        # Delete temporary file
        os.remove(audio_path)
        
        return jsonify({
            "audio": audio_data,
            "content_type": "audio/mp3"
        })
        
    except Exception as e:
        logger.error(f"TTS API error: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Run the Flask app
if _name_ == "_main_":
    # Make sure required packages are installed
    try:
        import time  # For file cleanup
        
        # Try to import gtts as fallback TTS option
        try:
            from gtts import gTTS
            logger.info("gTTS module available as fallback TTS")
        except ImportError:
            logger.warning("gTTS not available. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "gtts"])
            from gtts import gTTS
            logger.info("gTTS installed as fallback TTS")
    except Exception as setup_error:
        logger.warning(f"Setup error: {setup_error}")
        
    app.run(host="0.0.0.0", port=8080, debug=True)
