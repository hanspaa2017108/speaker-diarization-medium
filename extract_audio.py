# use this python file to extract audio if you have a video file

import subprocess

input_file = "/inputs/videos/video-2.mp4"
output_file = "/inputs/audios/audio-2.wav"

def extract_audio(input_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_file,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Audio extracted to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        exit(1)

extract_audio(input_file, output_file)