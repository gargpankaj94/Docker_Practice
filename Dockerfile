FROM python:3.7

CMD mkdir /hello_world
COPY . /hello_world

WORKDIR /hello_world

RUN pip3 install -r requirements.txt

CMD Hello_World.py
