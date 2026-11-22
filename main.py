import os
import time
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.textanalytics import TextAnalyticsClient
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

app = FastAPI(title="AI Document & Lecture Summarizer")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Azure Clients ---
blob_service = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
container_name = os.getenv("AZURE_STORAGE_CONTAINER", "uploads")

doc_client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOC_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOC_KEY")),
)

language_client = TextAnalyticsClient(
    endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_LANGUAGE_KEY")),
)


def upload_to_blob(file_bytes: bytes, filename: str):
    blob_client = blob_service.get_blob_client(container=container_name, blob=filename)
    blob_client.upload_blob(file_bytes, overwrite=True)


def extract_text_from_doc_or_image(file_bytes: bytes) -> str:
    poller = doc_client.begin_analyze_document(
        "prebuilt-read",
        body=file_bytes,
        content_type="application/octet-stream",
    )
    result = poller.result()
    return result.content or ""


def transcribe_audio_wav(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        transcript_parts = []
        done = False

        def handle_recognized(evt):
            if evt.result.text:
                transcript_parts.append(evt.result.text)

        def stop_cb(evt):
            nonlocal done
            done = True

        recognizer.recognized.connect(handle_recognized)
        recognizer.session_stopped.connect(stop_cb)
        recognizer.canceled.connect(stop_cb)

        recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.3)
        recognizer.stop_continuous_recognition()

        return " ".join(transcript_parts)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_summary_and_keywords(text: str, length: str = "detailed"):
    # Azure Language Abstractive Summary supports up to 125,000 chars
    full_text = text.strip()[:45000]
    documents = [full_text]

    sentence_map = {"short": 5, "medium": 9, "detailed": 15}
    sentence_count = sentence_map.get(length, 15)

    # 1. Detailed Abstractive Summary
    poller = language_client.begin_abstract_summary(
        documents, sentence_count=sentence_count
    )
    summary_results = poller.result()
    summary_text = ""
    for result in summary_results:
        if not result.is_error:
            summary_text = "\n\n".join([s.text for s in result.summaries])
        else:
            summary_text = f"Summary error: {result.error.message}"

    # 2. Key Bullet Highlights (Extractive Summary)
    highlights = []
    try:
        ext_poller = language_client.begin_extract_summary(
            documents, max_sentence_count=6
        )
        for res in ext_poller.result():
            if not res.is_error:
                highlights = [s.text for s in res.sentences]
    except Exception:
        highlights = []

    # 3. Key Phrases (Azure limit is 5,120 chars per doc)
    key_phrases = []
    try:
        kp_response = language_client.extract_key_phrases([full_text[:5000]])[0]
        if not kp_response.is_error:
            key_phrases = kp_response.key_phrases[:15]
    except Exception:
        key_phrases = []

    return summary_text, highlights, key_phrases


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")


@app.post("/summarize")
async def summarize_file(
    file: UploadFile = File(...),
    length: str = Form("detailed"),
):
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_docs = {".pdf", ".png", ".jpg", ".jpeg"}
    allowed_audio = {".wav"}

    if ext not in allowed_docs and ext not in allowed_audio:
        raise HTTPException(
            status_code=400,
            detail="Sirf PDF, JPG, PNG, aur WAV audio files supported hain.",
        )

    file_bytes = await file.read()

    # 1. Save to Azure Blob Storage
    upload_to_blob(file_bytes, file.filename)

    # 2. Extract Text
    if ext in allowed_docs:
        extracted_text = extract_text_from_doc_or_image(file_bytes)
        source_type = "Document / Image (Azure Doc Intelligence)"
    else:
        extracted_text = transcribe_audio_wav(file_bytes)
        source_type = "Lecture Audio (Azure AI Speech)"

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="File se koi text extract nahi ho paya.")

    # 3. Summarize with Azure AI Language
    summary, highlights, key_phrases = generate_summary_and_keywords(extracted_text, length)

    orig_words = len(extracted_text.split())
    sum_words = len(summary.split())

    return {
        "filename": file.filename,
        "source_type": source_type,
        "original_words": orig_words,
        "summary_words": sum_words,
        "summary": summary,
        "highlights": highlights,
        "key_phrases": key_phrases,
        "extracted_text": extracted_text[:12000],
    }