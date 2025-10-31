import os
import warnings
import logging

# --- Disable MediaPipe / TensorFlow Lite / OpenGL logs ---
os.environ["GLOG_minloglevel"] = "3"            # Suppress all GLOG logs (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"         # Hide TensorFlow INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # Optional: disable OneDNN verbose logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # Optional: hide GPU info
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # Prevent HuggingFace fork warnings

# --- Silence Python warnings and logging ---
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import librosa
import speech_recognition as sr
from deepface import DeepFace
import nltk
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import moviepy.editor as mp_edit # Importing moviepy.editor at the top for cleaner code

# === Setup and downloads ===
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load models
speech_recognizer = sr.Recognizer()
context_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Core function ===
def analyze_confidence(video_path, audio_file, question_text=""):
    # Initialize mediapipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    cap = cv2.VideoCapture(video_path)
    
    emotion_scores = {'happy': 0, 'neutral': 0, 'angry': 0, 'fear': 0, 'sad': 0, 'surprise': 0}
    total_frames = 0
    steady_eye_frames = 0
    moving_eye_frames = 0
    
    prev_left_eye = None
    prev_right_eye = None
    
    # ===================== Facial Modality =====================
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        # Emotion analysis every 20th frame
        if total_frames % 20 == 0:
            try:
                emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant = emotion_result[0]['dominant_emotion']
                if dominant in emotion_scores:
                    emotion_scores[dominant] += 1
            except Exception:
                pass
        
        # Eye movement detection
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in [33, 133, 159, 145]])
            right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in [362, 263, 386, 374]])
            
            if prev_left_eye is not None:
                left_diff = np.linalg.norm(left_eye - prev_left_eye)
                right_diff = np.linalg.norm(right_eye - prev_right_eye)
                
                if left_diff < 0.002 and right_diff < 0.002:
                    steady_eye_frames += 1
                else:
                    moving_eye_frames += 1
            
            prev_left_eye, prev_right_eye = left_eye, right_eye

    cap.release()
    face_mesh.close()
    
    # Normalize facial data
    total_emotions = sum(emotion_scores.values()) or 1
    happy_neutral_ratio = (emotion_scores['happy'] + emotion_scores['neutral']) / total_emotions
    negative_ratio = (emotion_scores['fear'] + emotion_scores['sad'] + emotion_scores['angry']) / total_emotions
    steady_ratio = steady_eye_frames / (steady_eye_frames + moving_eye_frames + 1e-6)
    
    facial_score = (0.6 * happy_neutral_ratio * 100 +
                    0.3 * steady_ratio * 100 -
                    0.2 * negative_ratio * 100)
    facial_score = np.clip(facial_score, 0, 100)
    
    # ===================== Speech Modality =====================
    # Extract audio from video
    try:
        video_clip = mp_edit.VideoFileClip(video_path)
        audio_path = audio_file
        video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        # PITCH ANALYSIS FIX: Use pyin for dynamic pitch range (variation)
        y, sr_audio = librosa.load(audio_path)
        
        # Use pyin for more reliable F0 estimation in speech
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), # Low Male (approx 65 Hz)
            fmax=librosa.note_to_hz('C5'), # High Female/Dynamic (approx 523 Hz)
            sr=sr_audio
        )
        
        # Filter for only voiced segments
        pitch_values = f0[voiced_flag]

        if len(pitch_values) > 10: # Require a minimum of 10 voiced frames for analysis
            # Measure Pitch Dynamism (Standard Deviation)
            pitch_std = np.std(pitch_values)
            
            # Normalization: A good dynamic range (std dev) for speech is around 30-50 Hz.
            MAX_DYNAMIC_RANGE_HZ = 80.0
            pitch_score = np.clip((pitch_std / MAX_DYNAMIC_RANGE_HZ) * 100, 0, 100)
        else:
            # Fallback if audio data is insufficient (too short or silent)
            pitch_score = 50 
        
        # Speech-to-text
        with sr.AudioFile(audio_path) as source:
            audio_data = speech_recognizer.record(source)
            try:
                transcribed_text = speech_recognizer.recognize_google(audio_data)
            except:
                transcribed_text = ""
    except Exception as e:
        print(f"Speech analysis failed: {e}")
        transcribed_text = ""
        pitch_score = 50 # Default score if audio extraction fails

    # Analyze filler words & stuttering
    fillers = ['um', 'uh', 'like', 'you know', 'actually', 'basically','ummmmm','uhhh','hmm','aaaaaaaaaaaaah','er','erm','huh']
    words = nltk.word_tokenize(transcribed_text.lower())
    filler_count = sum(words.count(f) for f in fillers)
    total_words = len(words) or 1
    filler_ratio = filler_count / total_words
    # Score decreases significantly for every 1% of words that are fillers
    filler_score = max(0, 100 - filler_ratio * 300) 
    
    # Combine speech features
    speech_score = np.clip(0.6 * filler_score + 0.4 * pitch_score, 0, 100)
    
    # ===================== Context Modality =====================
    if question_text and transcribed_text:
        q_emb, a_emb = context_model.encode([question_text, transcribed_text], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(q_emb, a_emb)
        context_score = float(similarity.item()) * 100
    else:
        context_score = 50  # neutral default
    
    # ===================== Final Confidence Calculation =====================
    confidence_score = np.clip(
        0.5 * facial_score + 0.3 * speech_score + 0.2 * context_score,
        0, 100
    )
    
    # ===================== Justification =====================
    justification = {
        "Happy/Neutral Expression %": round(happy_neutral_ratio * 100, 2),
        "Negative Emotion %": round(negative_ratio * 100, 2),
        "Eye Steadiness %": round(steady_ratio * 100, 2),
        "Pitch Score": round(pitch_score, 2), # Now reflects dynamic range, not high pitch
        "Filler Word Score": round(filler_score, 2),
        "Speech Score": round(speech_score, 2),
        "Context Similarity Score": round(context_score, 2),
        "Final Confidence Score": round(confidence_score, 2)
    }
    
    return justification

def get_score():
    question_text = ["What excites you most about this opportunity?", "Describe a challenge you overcame.", "What are your strongest skills?", "Where do you see yourself in 3 years?", "Why should we hire you?"]

    results = dict()

    for i in range(1, 6):
        video_path = f"./uploads/answer_{i}.webm"
        audio_file = f"./audios/answer_{i}.wav"
        print(f"\nAnalyzing {video_path}\n")
        # The pitch score generated here will now be more realistic (not stuck at 100)
        result = analyze_confidence(video_path, audio_file, question_text[i - 1])
        
        results[question_text[i - 1]] = result

    return results 