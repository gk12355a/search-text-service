# Sử dụng Python 3.11 slim giống với service AI của bạn cho đồng bộ
FROM python:3.11-slim

WORKDIR /app

# Cài đặt gcc và các gói c++ build tools để có thể build thư viện (như chromadb hay thefuzz)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tận dụng cache của Docker cho các layers pip install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code vào bên trong image
COPY . .

EXPOSE 8006

# Start fastapi server, port 8006 như config yaml đã map
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
