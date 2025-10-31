"""
This script extracts audio from a video file using the moviepy library.
It automatically handles file paths based on a simple folder structure.

Prerequisites:
- Install moviepy: pip install moviepy

Usage:
1. Create a subfolder named 'media_files' in the same directory as this script.
2. Place your video file (e.g., 'recorded_answer.avi') inside the 'media_files' folder.
3. Run the script from your terminal: python audio_extractor.py
"""

from moviepy.editor import VideoFileClip
import os
import sys

def extract_audio_from_video(video_path, output_path):
    """
    Extracts audio from a video file and saves it to a new file.

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path where the output audio file will be saved.
    """
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' was not found.")
        print("Please ensure your video is in the 'media_files' folder.")
        sys.exit(1)

    try:
        # Load the video clip
        print(f"Loading video from: {video_path}")
        video_clip = VideoFileClip(video_path)

        # Extract the audio
        audio_clip = video_clip.audio

        # Write the audio to a new file
        print(f"Writing audio to: {output_path}")
        audio_clip.write_audiofile(output_path)

        # Close the clips to free up resources
        video_clip.close()
        audio_clip.close()

        print("\nAudio extraction complete!")
        print(f"Audio saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the video file format is supported and the necessary codecs are installed (ffmpeg).")
        print("Moviepy automatically uses ffmpeg, so ensure it's in your system PATH or accessible.")

if __name__ == "__main__":
    # Get the script's directory to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the input video and output audio file paths
    # The script now assumes the video is in a 'media_files' subfolder
    input_video_path = os.path.join(script_dir, "media_files", "recorded_answer.mp4") 
    
    # The output audio will be saved in the same directory as the script
    output_audio_path = os.path.join(script_dir, "recorded_answer.mp3")

    extract_audio_from_video(input_video_path, output_audio_path)
