FROM python:3.7

CMD mkdir /hello_world
COPY . /hello_world

WORKDIR /hello_world

RUN pip3 install -r requirements.txt
RUN pip install --upgrade pip

RUN apt update

RUN apt install -y default-jre

RUN wget https://dlcdn.apache.org/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz

RUN tar -zxvf spark-3.3.2-bin-hadoop3.tgz

RUN mv spark-3.3.2-bin-hadoop3 /home/ubuntu

RUN export SPARK_HOME=/home/ubuntu/spark-3.3.2-bin-hadoop3
RUN export PATH=$SPARK_HOME/bin:$PATH
RUN export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH

CMD ["python3","Hello_World.py"]
