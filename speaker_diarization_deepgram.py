import os, mimetypes, requests
from dotenv import load_dotenv
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # set this in your env

def transcribe_with_diarization_local(path, model="nova-3"):  # or "nova-2"
    mime = mimetypes.guess_type(path)[0] or "audio/wav"
    with open(path, "rb") as f:
        audio_bytes = f.read()

    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": model,
        "diarize": "true",
        "utterances": "true",
        "smart_format": "true",  # punctuation, casing, etc.
        # "language": "en",  # set if needed
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mime,
    }
    resp = requests.post(url, params=params, headers=headers, data=audio_bytes, timeout=600)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    dg = transcribe_with_diarization_local("inputs/audios/audio-2.wav")

    # after you get `dg = transcribe...`
    alts = dg["results"]["channels"][0]["alternatives"][0]
    words = alts.get("words", [])
    print("words[0..5]:", [(w.get("word"), w.get("speaker")) for w in words[:6]])

    # Pretty print utterances with speakers
    for utt in dg.get("results", {}).get("utterances", []):
        s, e = utt["start"], utt["end"]
        spk = utt.get("speaker", "SPEAKER_00")
        print(f"[{s:7.2f}–{e:7.2f}] {spk}: {utt['transcript']}")
