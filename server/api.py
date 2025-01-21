import string

from flask import Flask, jsonify, request, Blueprint
import PyPDF2
ALLOWED_EXTENSIONS = {'pdf'}
from ai.similarity_scorer import SimilarityScorer
from server.async_processor import ResumeProcessor, process_to_pdf
import time
import random
import hashlib

app = Blueprint('resume-service', __name__)


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/v1/resume-service', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    keywords = request.form.getlist('keyword')
    if not keywords:
        return jsonify({'error': 'No keywords selected for uploading'}), 400

    if file and allowed_file(file.filename):
        processor = ResumeProcessor()
        item_hash = generate_unique_id()
        print('submitting')
        processor.submit(process_to_pdf(file), keywords, str(item_hash))

        return jsonify({'hash': item_hash}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400



@app.route('/api/v1/similarity-score', methods=['GET'])
def similarity_score():
    item_hash = request.args['applicant-hash']
    processor = ResumeProcessor()
    result = processor.check_item_status(item_hash)
    if result is not None:
        return jsonify({'hash': item_hash, 'result': str(processor.get_result(item_hash))}), 200

@app.route('/api/v1/check-status', methods=['GET'])
def check_status():
    item_hash = request.args['applicant-hash']
    processor = ResumeProcessor()
    result = processor.check_item_status(item_hash)
    if result is None:
        return jsonify({'result': 'Item does not exist.'}), 200
    return jsonify({'result': result}), 200


def generate_unique_id():
    # Define the character pool (alphanumeric)
    char_pool = string.ascii_letters + string.digits

    # Generate the first 4 characters
    part1 = ''.join(random.choices(char_pool, k=4))
    # Generate the next 3 characters
    part2 = ''.join(random.choices(char_pool, k=3))
    # Generate the final 2 characters
    part3 = ''.join(random.choices(char_pool, k=2))

    # Combine parts with dashes
    unique_id = f"{part1}-{part2}-{part3}"
    return unique_id
