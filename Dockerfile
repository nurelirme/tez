FROM tiagoprn/pythonocc-docker:latest

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install streamlit pandas numpy scikit-learn matplotlib

CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
