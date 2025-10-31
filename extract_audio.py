import complete_workflow as cf
import subprocess

def convert_to_vosk_format(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ac', '1',  # Mono channel
        '-ar', '16000', # Sample rate (often 16kHz for Vosk)
        '-acodec', 'pcm_s16le', # 16-bit PCM little-endian
        output_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully converted '{input_file}' to '{output_file}' for Vosk.")
        return "Done"
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
        return e
    
def extract_audios():
    for i in range(1, 6):
        print(f"Extracting Audio From - answer_{i}.webm")

        input_file = './uploads/answer_' + str(i) + '.webm'
        output_file = './audios/answer_' + str(i) + '.wav'

        cf.extract_audio_from_video(input_file, output_file) 