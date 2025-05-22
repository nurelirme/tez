FROM python:3.8

# Gerekli sistem kütüphanelerini yükle
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    build-essential \
    && pip install --upgrade pip

# Uygulama klasörüne geç
WORKDIR /app

# Tüm dosyaları konteynıra kopyala
COPY . /app

# Gerekli Python kütüphanelerini kur
RUN pip install -r requirements.txt

# Streamlit uygulamasını başlat
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
