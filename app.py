from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import requests
import time
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nlp = spacy.load('en_core_web_lg')

API_KEY = '84a1861a6a71472fba9d1f8d45266dba'

def upload_audio(file_path):
    headers = {
        'authorization': API_KEY
    }
    response = requests.post(
        'https://api.assemblyai.com/v2/upload',
        headers=headers,
        data=open(file_path, 'rb')
    )
    return response.json()['upload_url']

def request_transcription(audio_url):
    endpoint = "https://api.assemblyai.com/v2/transcript"
    json = {
        "audio_url": audio_url,
        "auto_highlights": True
    }
    headers = {
        "authorization": API_KEY,
        "content-type": "application/json"
    }
    response = requests.post(endpoint, json=json, headers=headers)
    return response.json()['id']

def get_transcription_result(transcript_id):
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {
        "authorization": API_KEY
    }
    while True:
        response = requests.get(endpoint, headers=headers)
        result = response.json()
        if result['status'] == 'completed':
            return result['text']
        elif result['status'] == 'failed':
            raise Exception("Transcription failed")
        time.sleep(5)

def transcribe_audio(file_path):
    audio_url = upload_audio(file_path)
    transcript_id = request_transcription(audio_url)
    text = get_transcription_result(transcript_id)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    transcribed_text = transcribe_audio(file_path)
    return jsonify({'transcription': transcribed_text, 'filename': filename})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text')
    length = int(data.get('length'))
    summary = generate_summary(text, length)
    return jsonify({'summary': summary})

def generate_summary(text, length):
    stopwords = STOP_WORDS

    # Tokenize text into sentences using spaCy
    doc = nlp(text)
    sentence_tokens = [sent.text for sent in doc.sents]

    # Tokenize each sentence and calculate word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text.lower() not in word_frequencies:
                word_frequencies[word.text.lower()] = 1
            else:
                word_frequencies[word.text.lower()] += 1

    # Normalize word frequencies by dividing by maximum frequency
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in nlp(sent):
            if word.text.lower() in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    # Select top sentences for the summary
    summary = nlargest(length, sentence_scores, key=sentence_scores.get)

    return ' '.join(summary)

if __name__ == '__main__':
    app.run(debug=True)
