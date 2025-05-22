FROM cadquery/pythonocc-core:latest

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    build-essential \
    && pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install streamlit pandas numpy scikit-learn matplotlib

CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
