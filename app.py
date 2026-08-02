
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
from yt_dlp.utils import DownloadError

# Whisper models are loaded lazily and cached so we don't reload per request.
_MODEL_CACHE = {}

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _MODEL_CACHE[name]


def _download_audio(url: str, workdir: str, cookies_file: str | None = None) -> str:
    """Download best audio from `url` into `workdir`, return the file path.

    `cookies_file` (or the YTDLP_COOKIES_FILE env var) can point to a
    Netscape-format cookies.txt file exported from a logged-in browser.
    This is required by sites like Instagram for posts/reels that return
    an empty response to anonymous requests.
    """
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

    cookies_file = cookies_file or os.getenv("YTDLP_COOKIES_FILE")
    if cookies_file:
        cookies_file = cookies_file.strip()

    used_auth = None
    if cookies_file:
        if not os.path.isfile(cookies_file):
            # Fail loudly instead of silently downloading without auth — a
            # typo'd/missing path used to look identical to "no cookies set".
            raise FileNotFoundError(
                f"Cookies file not found: {cookies_file!r}. Check the path is "
                "correct and readable from where the app is running."
            )
        ydl_opts["cookiefile"] = cookies_file
        used_auth = f"cookies file '{cookies_file}'"

    cookies_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser.strip(),)
        used_auth = used_auth or f"cookies from browser '{cookies_browser}'"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except DownloadError as e:
        message = str(e)
        if "instagram" in url.lower() and "empty media response" in message.lower():
            auth_note = (
                f"Tried with {used_auth}, but Instagram still refused it. The "
                "cookies may be expired/invalid, or this specific post requires "
                "additional verification (e.g. it's from an account you don't "
                "follow, or is restricted in your region). Try re-exporting "
                "fresh cookies right before downloading, or open the post's "
                "direct URL in the same logged-in browser first to confirm you "
                "can view it there."
                if used_auth
                else
                "Instagram returned an empty response for this post. It is likely "
                "private, age-restricted, or otherwise requires being logged in. "
                "Provide a cookies.txt file (export it from a browser where you're "
                "logged into Instagram) via the 'Cookies file' field or the "
                "YTDLP_COOKIES_FILE environment variable, then try again."
            )
            raise RuntimeError(auth_note) from e
        raise

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


def transcribe(url: str, model_name: str, cookies_file: str = "", progress=gr.Progress()):
    """Main pipeline: download -> transcribe -> return text + file paths."""
    if not url or not url.strip():
        raise gr.Error("Please paste a video link first.")

    url = url.strip()
    workdir = tempfile.mkdtemp(prefix="vt_")

    progress(0.1, desc="Downloading audio…")
    try:
        audio_path = _download_audio(url, workdir, cookies_file=cookies_file)
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

    with gr.Accordion("Advanced: private / login-required posts (e.g. Instagram)", open=False):
        cookies_in = gr.Textbox(
            label="Cookies file path",
            placeholder="/path/to/cookies.txt",
            info=(
                "Some sites (like Instagram) return an empty response for posts that "
                "require being logged in. Export cookies.txt from a browser session "
                "where you're logged in (e.g. with a 'Get cookies.txt' extension) and "
                "provide the path here, or set the YTDLP_COOKIES_FILE environment "
                "variable."
            ),
        )

    text_out = gr.HTML(label="Transcript (Bullet Points)")
    with gr.Row():
        txt_file = gr.File(label="Download .txt")
        srt_file = gr.File(label="Download .srt")

    go_btn.click(
        fn=transcribe,
        inputs=[url_in, model_in, cookies_in],
        outputs=[text_out, txt_file, srt_file],
    )
    url_in.submit(
        fn=transcribe,
        inputs=[url_in, model_in, cookies_in],
        outputs=[text_out, txt_file, srt_file],
    )

if __name__ == "__main__":
    # Cloud Run requires the app to bind to 0.0.0.0 and the provided PORT.
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "8080")))
    demo.launch(server_name=host, server_port=port, share=True)