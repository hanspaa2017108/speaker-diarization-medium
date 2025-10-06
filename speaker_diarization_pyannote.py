# speaker_diarization_pyannote.py
import os, json
from collections import defaultdict

import torch
import soundfile as sf
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Annotation

# === config ===
AUDIO_IN = "inputs/audios/audio-2.wav"     # mono 16k WAV recommended
RTTM_OUT = "outputs/audios/pyannote/audio-2.rttm"
RTTM_EXCL = "outputs/audios/pyannote/audio-2_exclusive.rttm"
MANIFEST_OUT = "outputs/audios/pyannote/manifest_2.json"
SPEAKER_TXT_DIR = "outputs/audios/pyannote/speakers_text_2"
MIN_SPK = 4
MAX_SPK = 6   # give headroom if clips may have >4 speakers
END_PAD = 0.12  # seconds to pad last diar turn to catch trailing ASR tails

# Whisper wrapper (renamed to avoid shadowing openai-whisper package)
from asr_whisper import transcribe_audio

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ============== DIARIZATION ==============
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

def diarize_audio(audio_file: str):
    # Load as waveform dict to avoid TorchCodec warnings
    wave, sr = sf.read(audio_file, dtype="float32")
    if wave.ndim == 1:
        wave = wave[None, ...]  # (1, time)
    wave_t = torch.from_numpy(wave)
    out = pipeline(
        {"waveform": wave_t, "sample_rate": sr},
        min_speakers=MIN_SPK,
        max_speakers=MAX_SPK
    )
    return out, sr

diar, sr = diarize_audio(AUDIO_IN)

# Write BOTH RTTMs (inclusive & exclusive) for inspection- RTTM file format: TYPE  FILE-ID  CHAN  START  DUR  ORTHO  STYPE  SPEAKER-ID  CONF  SLAT
# TYPE: SPEAKER, FILE-ID: 1, CHAN: 1, START: 0.031, DUR: 0.354, ORTHO: <NA>, STYPE: <NA>, SPEAKER-ID: SPEAKER_02, CONF: <NA>, SLAT: <NA>
ann_inclusive: Annotation = diar.speaker_diarization
ann_exclusive: Annotation = diar.exclusive_speaker_diarization

os.makedirs(os.path.dirname(RTTM_OUT), exist_ok=True)
os.makedirs(SPEAKER_TXT_DIR, exist_ok=True)

with open(RTTM_OUT, "w") as f:
    ann_inclusive.write_rttm(f)
with open(RTTM_EXCL, "w") as f:
    ann_exclusive.write_rttm(f)

# Print segments (debug) from exclusive (cleaner, no overlaps)
for seg, _, spk in ann_exclusive.itertracks(yield_label=True):
    print(f"{spk}: {seg.start:.2f}s → {seg.end:.2f}s (dur={(seg.end - seg.start):.2f}s)")

# Build diarization turns from EXCLUSIVE annotation
turns = [
    {"speaker": spk, "start": float(seg.start), "end": float(seg.end)}
    for seg, _, spk in ann_exclusive.itertracks(yield_label=True)
]
turns.sort(key=lambda x: x["start"])
# pad last turn a bit to catch trailing ASR tails
if turns:
    turns[-1]["end"] += END_PAD

# ============== TRANSCRIPTION (Whisper) ==============
asr = transcribe_audio(
    AUDIO_IN,
    model_name="large-v3",  # force strong model; your wrapper falls back if needed
    temperature=0.0,
    best_of=1,
)
print("ASR model used:", asr.get("model_used"))

# ============== FUSION: split ASR segments at diar boundaries ==============
def split_asr_by_turns(asr_seg, sorted_turns):
    """Split one ASR segment wherever diarization turns change speaker."""
    a0, a1 = asr_seg["start"], asr_seg["end"]
    pieces = []
    for t in sorted_turns:
        if t["end"] <= a0:
            continue
        if t["start"] >= a1:
            break
        s = max(a0, t["start"])
        e = min(a1, t["end"])
        if e > s:
            pieces.append({
                "speaker": t["speaker"],
                "start": s,
                "end": e,
                "duration": round(e - s, 3),
                "text": asr_seg["text"],  # (optional) later: slice words w/ aligner
            })
    return pieces

fused_segments = []
for seg in asr["segments"]:
    parts = split_asr_by_turns(seg, turns)
    if parts:
        fused_segments.extend(parts)
    else:
        # If no overlap with any turn, keep as UNKNOWN
        fused_segments.append({
            "speaker": "UNKNOWN",
            "start": seg["start"],
            "end": seg["end"],
            "duration": round(seg["end"] - seg["start"], 3),
            "text": seg["text"],
        })

# Per-speaker rollups
speaker_text = defaultdict(list)
speaker_dur = defaultdict(float)
for seg in fused_segments:
    spk = seg["speaker"]
    speaker_text[spk].append(seg["text"])
    speaker_dur[spk] += seg["duration"]

# Save per-speaker text (handy QA / prompts)
for spk, lines in speaker_text.items():
    fname = spk.replace(":", "_")
    with open(os.path.join(SPEAKER_TXT_DIR, f"{fname}.txt"), "w") as f:
        f.write((" ".join(lines)).strip() + "\n")

# Manifest (speaker-attributed transcript + diarization)
manifest = {
    "input": AUDIO_IN,
    "sample_rate": sr,
    "asr": {
        "model_used": asr["model_used"],
        "language": asr["language"],
        "full_text_preview": (asr["text"] or "")[:200],
    },
    "speakers": [
        {"id": spk, "total_sec": round(dur, 3)}
        for spk, dur in sorted(speaker_dur.items(), key=lambda x: x[0])
    ],
    "segments": fused_segments,  # <- who said what, when (after splitting on turn boundaries)
}

with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved:\n- {RTTM_OUT}\n- {RTTM_EXCL}\n- {MANIFEST_OUT}\n- {SPEAKER_TXT_DIR}/*.txt")
