# 1. Base image
FROM python:3.12-slim

# 2. Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies (only if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install uv
RUN pip install --no-cache-dir uv

# 6. only Flask app requirements first (better caching)
COPY flask_app/requirements.txt ./requirements.txt

# 7. Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

# 8. Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# 9. Copy application code
COPY flask_app/ .

# 10. Copy model artifact
COPY models/vectorizer.pkl ./models/vectorizer.pkl

# 11. Expose port
EXPOSE 5000

# local
# CMD ["python", "app.py"] 

# 12. Run app (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]