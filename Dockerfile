FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
# RUN python -m nltk.downloader punkt wordnet stopwords averaged_perceptron_tagger

EXPOSE 5005

# Change to use Flask development server temporarily for testing
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
CMD ["flask", "run", "--host=0.0.0.0", "--port=5005"]