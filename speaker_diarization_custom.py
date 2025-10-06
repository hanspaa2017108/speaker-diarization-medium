import librosa
import traceback
from faster_whisper import WhisperModel
import torch
import datetime
from pathlib import Path
import pandas as pd
import time
import os
import numpy as np
import speechbrain
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

audio_file_path = "inputs/audios/audio-2.wav"

whisper_models = ["base", "medium", "large-v1", "large-v2", "large-v3"]

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def speech_to_text(audio_file, whisper_model):
    model = WhisperModel(whisper_model, compute_type="int8")
    time_start = time.time()

    try:
        # get duration
        audio_data, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio_data) / sample_rate

        # transcribe audio
        options = dict(language='en', beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file, **transcribe_options)

        #convert back to original openai format
        segments = []
        for segment_chunk in segments_raw:
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            segments.append(chunk)
        
    except Exception as e:
        raise RuntimeError("Error Converting Video to Audio")

    try:
        # create embedding
        def segment_embedding(segment):
            try:
                audio = Audio()
                start = segment["start"]
                end = min(duration, segment["end"])

                clip = Segment(start, end)
                waveform, sample_rate = audio.crop(audio_data, clip)

                embeddings = embedding_model(waveform[None])

                return embeddings

            except Exception as e:
                traceback.print_exc()
                raise RuntimeError("Error during Segment Embedding", e)

        # create embeddings
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)

        # Assign speaker label
        best_num_speaker = 2
        clustering = AgglomerativeClustering(best_num_speaker). fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER' + str(labels[i] + 1)

        # create output
        objects = {
            'Start' : [],
            'End' : [],
            'Speaker' : [],
        }
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['End'].append(str(convert_time(segments[-1]["end"])))
        
        save_path = os.path.join("outputs", "audios", "custom", f"{os.path.basename(audio_file).split('.')[0]}_{whisper_model}.csv")
        df_results = pd.DataFrame(objects)
        df_results.to_csv(save_path)
        return df_results, save_path

    except Exception as e:
        print("Exception: ", e)
        raise RuntimeError("Error during Speaker Diarization", e)

audio_file = "inputs/audios/audio-2.wav"
selected_whisper_model = "large-v3"

transcription_results, save_path = speech_to_text(audio_file, selected_whisper_model)
print(transcription_results)
print(save_path)