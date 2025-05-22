FROM python:3.8

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    libx11-dev \
    libglu1-mesa-dev \
    libxi-dev \
    libxmu-dev \
    build-essential \
    && pip install --upgrade pip

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
 
