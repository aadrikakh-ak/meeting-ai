from flask import Flask, render_template, request
import whisper
from transformers import pipeline
import os

app = Flask(__name__)

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading summarizer...")
summarizer = pipeline(
    "text-generation",
    model="gpt2"
)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():

    transcript = ""
    summary = ""
    action_items = []

    if request.method == "POST":

        audio = request.files["audio"]

        if audio:

            filepath = os.path.join(
                UPLOAD_FOLDER,
                audio.filename
            )

            audio.save(filepath)

            print("Transcribing audio...")

            result = whisper_model.transcribe(filepath)

            transcript = result["text"]

            print("Generating summary...")

            summary_result = summarizer(
                "Summarize this meeting: " + transcript,
                max_length=100,
                do_sample=False
            )

            summary = summary_result[0]["generated_text"]

            lines = transcript.split(".")

            keywords = [
                "will",
                "must",
                "should",
                "need to",
                "assigned"
            ]

            for line in lines:

                for keyword in keywords:

                    if keyword in line.lower():

                        action_items.append(
                            line.strip()
                        )

                        break

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary,
        action_items=action_items
    )

if __name__ == "__main__":
    app.run(debug=True)