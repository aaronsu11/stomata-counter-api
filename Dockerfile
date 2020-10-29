FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim

COPY . /app

