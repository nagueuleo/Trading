FROM python:3.12-alpine


WORKDIR /Trading

COPY . .


ENV FLASK_APP=main.py FLASK_RUN_HOST=0.0.0.0

#COPY requirements.txt requirements.txt 
# Installer les outils de construction nécessaires
RUN apk add --no-cache build-base

RUN pip install -r requirements.txt

EXPOSE 5000



CMD ["flask", "run", "--host=0.0.0.0"]