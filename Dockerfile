FROM python:3.10-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN pip freeze > /requirements_freeze.txt

# ------------------------------

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /requirements_freeze.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements_freeze.txt

COPY . .

EXPOSE 5001

CMD ["python", "server.py"]
