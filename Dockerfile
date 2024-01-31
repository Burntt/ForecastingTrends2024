FROM python:3.8

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
COPY ./exp ./exp
COPY ./models ./models
COPY ./utils ./utils
COPY ./data ./data
COPY ./scripts ./scripts
COPY ./main_informer_mlflow.py ./main_informer_mlflow.py

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/app/main_informer_mlflow.py"]