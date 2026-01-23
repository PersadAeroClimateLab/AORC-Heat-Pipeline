FROM python:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "run.py"]