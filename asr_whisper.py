# asr_whisper.py
import torch

def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3",  # demand large-v3 by default
    language: str | None = None,
    device: str | None = None,
    fp16: bool | None = None,
    **whisper_kwargs,
):
    """
    Returns:
      {
        "text": str,
        "segments": [{"start": float, "end": float, "text": str}],
        "language": str | None,
        "model_used": str,
      }
    """
    try:
        import whisper  # openai-whisper
    except Exception as e:
        raise RuntimeError(
            "openai-whisper not installed in this env. Run:\n"
            "  python -m pip install -U openai-whisper"
        ) from e

    # choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = device == "cuda"

    # try EXACT model; if it fails, raise with a clear message
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Whisper model '{model_name}'.\n"
            "Likely causes:\n"
            " - Your openai-whisper version is old (upgrade it)\n"
            " - ffmpeg missing from PATH (install it)\n"
            "Fix then re-run. Original error:\n"
            f"{type(e).__name__}: {e}"
        ) from e

    params = dict(
        language=language,
        temperature=0.0,
        best_of=1,
        beam_size=None,
        fp16=fp16,
        verbose=False,
    )
    params.update({k: v for k, v in whisper_kwargs.items() if v is not None})

    result = model.transcribe(audio_path, **params)

    segments = [
        {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": (seg.get("text") or "").strip(),
        }
        for seg in result.get("segments", [])
    ]

    return {
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "language": result.get("language"),
        "model_used": model_name,
    }
