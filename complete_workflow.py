from moviepy.editor import VideoFileClip
import os
import speech_recognition as sr
from pydub import AudioSegment
import sys
import subprocess 
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json

# Suppress Vosk logging for cleaner output (optional)
SetLogLevel(-1)

# Path to your downloaded Vosk model
model_path = "./vosk-model-en-us-0.42-gigaspeech" 

# Load the Vosk model
model = Model(model_path)

def extract_audio_from_video(video_path, output_path):
    """
    Extracts audio from a video file and saves it to a new file 
    using a direct FFmpeg call via subprocess to ensure precise parameters.

    Parameters:
    - Mono channel: -ac 1
    - Sample rate: 16000 Hz (-ar 16000)
    - Codec: pcm_s16le (-acodec pcm_s16le)

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path where the output audio file will be saved (e.g., must be a .wav file).
    """
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' was not found.")
        return

    # Define the required FFmpeg command and arguments
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_path,         # Input file
        '-vn',                    # Disable video recording
        '-ac', '1',               # Mono channel
        '-ar', '16000',           # Sample rate (16kHz)
        '-acodec', 'pcm_s16le',   # 16-bit PCM little-endian codec
        '-y',                     # Overwrite output file if it exists
        output_path               # Output file path (should be .wav)
    ]

    print(f"Loading video from: {video_path}")
    print("Bypassing moviepy for conversion to ensure exact parameters using subprocess...")
    print(f"Executing FFmpeg command: {' '.join(ffmpeg_command)}")

    try:
        # Execute the FFmpeg command
        # subprocess.run is used to execute the external command and wait for it to complete
        result = subprocess.run(
            ffmpeg_command,
            check=True,  # Raise a CalledProcessError if the command returns a non-zero exit code
            capture_output=True,
            text=True
        )

        if os.path.exists(output_path):
            print("\nAudio extraction and conversion complete!")
            print(f"Audio saved to: {output_path}")
            # Optionally, print FFmpeg's output for debugging
            # print("FFmpeg Output:\n", result.stdout)
        else:
             print("\nError: FFmpeg executed, but the output file was not created.")
             print("FFmpeg Error Output:\n", result.stderr)
             
    except FileNotFoundError:
        print("\nFATAL ERROR: FFmpeg executable not found.")
        print("Please ensure FFmpeg is installed and the 'ffmpeg' command is accessible in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred during FFmpeg execution (Exit Code {e.returncode}):")
        print("FFmpeg Error Output:\n", e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 


# --- IMPORTANT: Configure these paths with the FULL PATH on your system ---
# Replace with the actual full path to your ffmpeg.exe file
# Example for Windows: r"C:\path\to\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
# Example for macOS/Linux: "/usr/local/bin/ffmpeg"
FFMPEG_PATH = "ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"

# Replace with the actual full path to your ffprobe.exe file
# Example for Windows: r"C:\path\to\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"
# Example for macOS/Linux: "/usr/local/bin/ffprobe"
FFPROBE_PATH = "ffmpeg-master-latest-win64-gpl-shared/bin/ffprobe.exe"
# -------------------------------------------------------------------------

# --- Configuration for the audio file ---
# Replace with the full path to your audio file
# Example: r"C:\Users\YourName\Documents\audio_files\output_audio.mp3"
# AUDIO_FILE_PATH = "audio_files\output_audio.mp3"
# ----------------------------------------


def check_paths():
    """Verifies that all required file paths exist before proceeding."""
    print("Checking file and dependency paths...")
    
    if not os.path.exists(FFMPEG_PATH):
        print(f"Error: ffmpeg executable not found at '{FFMPEG_PATH}'.")
        print("Please update the FFMPEG_PATH variable with the full, correct path.")
        sys.exit(1)

    if not os.path.exists(FFPROBE_PATH):
        print(f"Error: ffprobe executable not found at '{FFPROBE_PATH}'.")
        print("Please update the FFPROBE_PATH variable with the full, correct path.")
        sys.exit(1)

    # if not os.path.exists(AUDIO_FILE_PATH):
    #     print(f"Error: The audio file was not found at '{AUDIO_FILE_PATH}'.")
    #     print("Please update the AUDIO_FILE_PATH variable with the full, correct path.")
    #     sys.exit(1)
    
    print("All paths are valid. Proceeding with transcription...")


def extract_text(audio_file):
    # Open the audio file
    wf = wave.open(audio_file, "rb") 

    # Check audio file properties
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format, mono, 16-bit PCM.")
        exit(1)

    # Create a recognizer object
    recognizer = KaldiRecognizer(model, wf.getframerate())

    # Process audio in chunks
    while True:
        data = wf.readframes(4000) # Read 4000 frames at a time
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            print(result["text"])

    # Get the final result for any remaining audio
    final_result = json.loads(recognizer.FinalResult())
    
    wf.close() 

    return final_result["text"] 