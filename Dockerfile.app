FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy frontend web application requirements
COPY Api/ ./Api/
COPY models/ ./models/
COPY Data/Processed/zameen_train_preprocessed.csv ./Data/Processed/zameen_train_preprocessed.csv

EXPOSE 8501

CMD ["streamlit", "run", "Api/app.py", "--server.port=8501", "--server.address=0.0.0.0"]