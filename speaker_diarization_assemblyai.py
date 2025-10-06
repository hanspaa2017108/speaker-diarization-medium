import assemblyai as aai
import os
from dotenv import load_dotenv

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# You can use a local filepath:
audio_file = "inputs/audios/audio-2.wav"

# # Or use a publicly-accessible URL:
# audio_file = (
#     "https://assembly.ai/wildfires.mp3"
# )

config = aai.TranscriptionConfig(
  speech_model=aai.SpeechModel.universal,
  speaker_labels=True,
  speakers_expected=4,
  #multichannel=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")

# # option to export SRT
# srt = transcript.export_subtitles_srt(
#   # Optional: Customize the maximum number of characters per caption
#   chars_per_caption=32
#   )
# with open(f"transcript_{transcript.id}.srt", "w") as srt_file:
#   srt_file.write(srt)