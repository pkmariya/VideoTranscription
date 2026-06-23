
"""
Video Transcriber — a Gradio app using faster-whisper.

Paste a video link (YouTube or most other sites), the app downloads the
audio with yt-dlp, transcribes it locally with faster-whisper, and shows
the transcript. You can also download the result as .txt or .srt.

Run:  python app.py
Then open the local URL it prints (default http://127.0.0.1:7860).
"""

import os
import tempfile
import datetime as _dt

import gradio as gr
from faster_whisper import WhisperModel
import yt_dlp

# Whisper models are loaded lazily and cached so we don't reload per request.
_MODEL_CACHE = {}

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _MODEL_CACHE[name]


def _download_audio(url: str, workdir: str) -> str:
    """Download best audio from `url` into `workdir`, return the file path."""
    out_template = os.path.join(workdir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Convert whatever we get into a wav Whisper/ffmpeg reads cleanly.
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    for f in os.listdir(workdir):
        if f.startswith("audio."):
            return os.path.join(workdir, f)
    raise FileNotFoundError("Audio download failed — no output file produced.")


def _format_timestamp(seconds: float) -> str:
    td = _dt.timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg.start)
        end = _format_timestamp(seg.end)
        text = seg.text.strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _format_as_bullets(segments) -> str:
    """Format transcript segments as bullet points without timestamps."""
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if text:  # Only add non-empty segments
            lines.append(f"• {text}")
    return "\n".join(lines)


def _format_as_html_bullets(segments) -> str:
    """Format transcript segments as HTML bullet points for display (no timestamps)."""
    lines = ["<ul style='line-height: 1.8; font-size: 16px;'>"]
    for seg in segments:
        text = seg.text.strip()
        if text:  # Only add non-empty segments
            lines.append(f"  <li>{text}</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def transcribe(url: str, model_name: str, progress=gr.Progress()):
    """Main pipeline: download -> transcribe -> return text + file paths."""
    if not url or not url.strip():
        raise gr.Error("Please paste a video link first.")

    url = url.strip()
    workdir = tempfile.mkdtemp(prefix="vt_")

    progress(0.1, desc="Downloading audio…")
    try:
        audio_path = _download_audio(url, workdir)
    except Exception as e:
        raise gr.Error(f"Could not download audio from that link: {e}")

    progress(0.4, desc=f"Loading Whisper '{model_name}' model…")
    model = _load_model(model_name)

    progress(0.6, desc="Transcribing (this can take a while)…")
    try:
        segments, info = model.transcribe(audio_path, language="en")
        segments = list(segments)  # Convert generator to list
    except Exception as e:
        raise gr.Error(f"Transcription failed: {e}")

    # Format transcript as bullet points
    bullet_text = _format_as_bullets(segments)
    html_bullets = _format_as_html_bullets(segments)

    # Write downloadable files.
    txt_path = os.path.join(workdir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(bullet_text)

    srt_path = os.path.join(workdir, "transcript.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_to_srt(segments))

    progress(1.0, desc="Done")
    return html_bullets, txt_path, srt_path


with gr.Blocks(title="Video Transcriber") as demo:
    gr.Markdown(
        "# 🎬 Video Transcriber\n"
        "Paste a video link (YouTube or most other sites). The app downloads "
        "the audio, transcribes it locally with faster-whisper, and shows the text as formatted bullet points."
    )

    with gr.Row():
        url_in = gr.Textbox(
            label="Video link",
            placeholder="https://www.youtube.com/watch?v=…",
            scale=4,
        )
        model_in = gr.Dropdown(
            choices=MODEL_CHOICES,
            value="base",
            label="Whisper model",
            info="Larger = more accurate but slower.",
            scale=1,
        )

    go_btn = gr.Button("Transcribe", variant="primary")

    text_out = gr.HTML(label="Transcript (Bullet Points)")
    with gr.Row():
        txt_file = gr.File(label="Download .txt")
        srt_file = gr.File(label="Download .srt")

    go_btn.click(
        fn=transcribe,
        inputs=[url_in, model_in],
        outputs=[text_out, txt_file, srt_file],
    )
    url_in.submit(
        fn=transcribe,
        inputs=[url_in, model_in],
        outputs=[text_out, txt_file, srt_file],
    )


if __name__ == "__main__":
    demo.launch()