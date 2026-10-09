FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# TensorFlow requires the OpenMP runtime on Debian-based images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

EXPOSE 5000

CMD ["python", "-c", "import os; from backend.app import app; app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False)"]
