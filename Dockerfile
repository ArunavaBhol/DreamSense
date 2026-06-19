FROM python:3.10-slim

WORKDIR /app

# Copy dependency configs first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ./src ./src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
