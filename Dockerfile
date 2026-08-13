FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

ENV CONTENT_AGENT_HOST=0.0.0.0
ENV CONTENT_AGENT_PORT=7860
ENV CONTENT_AGENT_DB_PATH=/data/content_agent.db

EXPOSE 7860

CMD ["python", "app.py"]
