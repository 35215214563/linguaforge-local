FROM python:3.11-slim

WORKDIR /app

COPY frontend/srt.html ./srt.html
COPY frontend/css ./css
COPY frontend/js ./js

RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app

EXPOSE 8000

USER app

CMD ["python3", "-m", "http.server", "8000", "--bind", "0.0.0.0"]
