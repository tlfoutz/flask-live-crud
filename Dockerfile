FROM python:3.6-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENV PORT 8080

CMD [ "flask", "run", "--host=0.0.0.0", "--port=8080"]
