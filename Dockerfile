# Production image for a Hugging Face Docker Space (free CPU). Spaces run as uid 1000 and
# route traffic to app_port 7860 (set in README.md's front matter).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/home/user/app
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user
COPY --chown=user medrag medrag
COPY --chown=user scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py
COPY --chown=user app.py .
COPY --chown=user .streamlit .streamlit

EXPOSE 7860
# XSRF protection breaks file uploads inside the Hugging Face iframe.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--client.showErrorDetails=none"]
