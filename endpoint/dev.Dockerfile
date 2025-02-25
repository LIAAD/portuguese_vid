FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip

RUN pip install -U pt-vid[demo]

RUN pip install -U spacy

COPY . .

EXPOSE 80

ENTRYPOINT ["fastapi", "run", "main.py", "--host=0.0.0.0", "--port=80"]
