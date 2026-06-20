# Video Transcriber

A small Gradio app: paste a video link, it downloads the audio, transcribes it
locally with [OpenAI Whisper](https://github.com/openai/whisper), and shows the
transcript. Download the result as `.txt` or `.srt`.

## Requirements

- **Python 3.9–3.11**
- **ffmpeg** (required by both yt-dlp and Whisper):
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from https://ffmpeg.org and add it to your PATH

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the chosen Whisper model weights automatically.

## Run

```bash
python app.py
```

Open the URL it prints (default http://127.0.0.1:7860), paste a link, pick a
model, and click **Transcribe**.

## Notes

- **Model size** trades speed for accuracy: `tiny`/`base` are fast, `small`/
  `medium`/`large` are slower but more accurate. A GPU helps a lot for the
  larger models; on CPU stick to `tiny`/`base`.
- Works with YouTube and most sites yt-dlp supports, plus direct video URLs.
- Only download/transcribe content you have the right to use.
- To share the app on a temporary public link, change the last line to
  `demo.launch(share=True)`.
