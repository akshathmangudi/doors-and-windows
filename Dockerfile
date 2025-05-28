# Use slim Python base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only relevant files
COPY requirements.txt .

# Install Python dependencies first (leverages Docker cache)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy the rest of the code (after installing deps for better caching)
COPY . .

# ✅ Explicitly copy the pretrained model weights if you're using them
COPY models/ models/

# Streamlit environment config
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port used by Streamlit
EXPOSE 7860

# Start the app
CMD ["streamlit", "run", "app.py"]
