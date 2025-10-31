from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json

# Suppress Vosk logging for cleaner output (optional)
SetLogLevel(-1)

# Path to your downloaded Vosk model
model_path = "./vosk-model-en-us-0.42-gigaspeech" 

# Load the Vosk model
model = Model(model_path)

# Open the audio file
wf = wave.open("./audios/answer_1.wav", "rb") 

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
print(final_result["text"])

wf.close() 