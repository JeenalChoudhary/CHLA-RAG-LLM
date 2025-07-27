FROM node:20-alpine AS builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/. .

RUN npm run build

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip3 install --no-cache-dir --timeout=600 -r requirements.txt

COPY --from=builder /app/frontend/dist ./static

COPY backend/. .

EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]