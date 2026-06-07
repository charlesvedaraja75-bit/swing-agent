FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

COPY swing_agent.py.

CMD ["python", "swing_agent.py"]
