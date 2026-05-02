FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    CONFIG_DIR=/app/configs \
    SESSION_OUTPUT_DIR=/app/outputs/sessions

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements ./requirements
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements/base.txt && \
    pip install -e .

COPY apps ./apps
COPY configs ./configs
COPY models ./models
COPY scripts ./scripts
COPY data ./data

EXPOSE 8501 8000

CMD ["streamlit", "run", "apps/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
