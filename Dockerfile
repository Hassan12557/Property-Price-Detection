FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project folder
COPY . .

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run Streamlit directly on the port Hugging Face expects
CMD ["streamlit", "run", "api/app.py", "--server.port=7860", "--server.address=0.0.0.0"]