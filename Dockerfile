FROM python:3.11-slim

# Tesseract OCR (+ English and orientation-detection data) for scanned statements
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# make sure a trained model exists inside the image
RUN test -f models/classifier.joblib || python -m bankparser.classify.train

ENV PORT=8501
EXPOSE 8501
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true
