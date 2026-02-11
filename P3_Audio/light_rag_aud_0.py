import numpy as np
from pydub import AudioSegment
from textblob import TextBlob
import speech_recognition as sr
import os
import io
import random

AudioSegment.ffmpeg = "C:\\Users\\malas\\shri_1000_names\\PythonProjects\\common_lib\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\Users\\malas\\shri_1000_names\\PythonProjects\\common_lib\\ffmpeg\\bin\\ffprobe.exe"
AudioSegment.converter = "C:\\Users\\malas\\shri_1000_names\\PythonProjects\\common_lib\\ffmpeg\\bin\\ffmpeg.exe"

class VoiceSentimentAnalyzer:
    def __init__(self, silence_threshold=-40.0):
        self.recognizer = sr.Recognizer()
        self.silence_thresh = silence_threshold

    def analyze_audio_features(self, audio_segment):
        """
        Extracts 'Light' prosodic features:
        1. Volume (dBFS): loudness.
        2. Pitch (Estimate): utilizing Zero-Crossing Rate (ZCR) or FFT as a proxy for agitation.
        """
        # 1. Volume Analysis (Loudness)
        avg_volume = audio_segment.dBFS

        # 2. Pitch/Agitation Estimate (using NumPy)
        # We convert audio to raw data to find frequency cues
        samples = np.array(audio_segment.get_array_of_samples())
        if len(samples) == 0:
            return -100, 0

        # Zero Crossing Rate (ZCR) is a lightweight proxy for "noise/agitation"
        # High ZCR often correlates with rough, angry speech.
        zcr = ((samples[:-1] * samples[1:]) < 0).sum() / len(samples)
        return avg_volume, zcr

    def extract_text(self, audio_path):
        """Transcribes audio to text using SpeechRecognition."""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                # Using Google's free API for 'Light' demo; replace with local Whisper for production
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            print(f"Error transcribing: {e}")
            return ""

    def process_recording(self, file_path):

        # 1. Load Audio with explicit format check
        try:
            new_path = file_path.replace(".wav", ".mp3")
            wav_audio_segment = AudioSegment.from_wav(file_path)
            memory_buffer = io.BytesIO()
            wav_audio_segment.export(memory_buffer, format="mp3")
            mp3_data_bytes = memory_buffer.getvalue()
            with open(new_path, "wb") as f:
                f.write(mp3_data_bytes)

            mp3_audio_segment = AudioSegment.from_mp3(new_path)

            # DEBUG: Check if audio actually contains data
            if len(mp3_audio_segment) == 0:
                print("Error: Audio file is empty (0 seconds).")
                return
        except FileNotFoundError:
            print("Error: File not found. Check the path.")
            return
        except Exception as e:
            print(f"Error loading audio. Is FFmpeg installed? Details: {e}")
            return

        # 2. Extract Acoustic Features (The "How" it was said)
        volume_db, agitation_score = self.analyze_audio_features(mp3_audio_segment)

        # 3. Extract Text & Sentiment (The "What" was said)
        text = self.extract_text(file_path)
        sentiment_score = 0
        if text:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity  # Range: -1 (Negative) to 1 (Positive)

        # 4. Light RAG Decision Logic
        # We define "Anger" as: (Loud Volume OR High Agitation) AND (Negative Text)
        is_loud = volume_db > -15.0  # Threshold depends on recording quality (standard is -20dBFS)
        is_agitated = agitation_score > 0.15  # High variance/frequency
        is_negative = sentiment_score < -0.1

        print("\n--- ANALYSIS REPORT ---")
        print(f"🔊 Volume: {volume_db:.2f} dB {'(LOUD)' if is_loud else ''}")
        print(f"📈 Agitation (ZCR): {agitation_score:.4f}")
        print(f"📝 Transcript: \"{text}\"")
        print(f"💡 Sentiment Polarity: {sentiment_score:.2f} {'(NEGATIVE)' if is_negative else ''}")

        # Final Verdict
        if (is_loud or is_agitated) and is_negative:
            print("\n🚨 ALERT: CUSTOMER IS ANGRY/FRUSTRATED 🚨")
            print("Reason: Negative sentiment detected combined with high volume/agitation.")
        elif is_negative:
            print("\n⚠️ Note: Customer is unhappy (Negative text), but tone is calm.")
        elif is_loud:
            print("\n⚠️ Note: Customer is loud, but words are not negative (Possible bad mic/background noise).")
        else:
            print("\n✅ Customer seems calm.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    analyzer = VoiceSentimentAnalyzer()

    # Analyse voice sentiment - Customer is Angry/ Calm / Normal
    for root, dirs, files in os.walk(os.path.abspath("data_in")):
        random.shuffle(files)
        idx = 0
        for one_file in files:
            if one_file.endswith(".wav"):
                idx+=1
                if idx > 4:
                    continue
                print("\n\n" + os.path.join(root, one_file))
                audio_file = os.path.join(root, one_file)
                analyzer.process_recording(audio_file)
