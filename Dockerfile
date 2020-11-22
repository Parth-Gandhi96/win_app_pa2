FROM ubuntu:latest

FROM python:3

RUN apt-get update
RUN apt install default-jdk -y

RUN java -version

RUN python -v

RUN command -v pip

RUN pip3 install pyspark
RUN pip3 install findspark

RUN pip install numpy

ENV EXPORT SPARK_HOME=/opt/spark

ENV EXPORT PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

ENV EXPORT PYSPARK_PYTHON=/usr/bin/python3

RUN java -version; javac -version; scala -version; git --version

RUN wget https://downloads.apache.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz

RUN tar xvf spark-*

RUN mv spark-3.0.1-bin-hadoop2.7 /opt/spark

COPY code_wine_predict_docker_1.py code_wine_predict_docker_1.py

COPY TrainingDataset.csv TrainingDataset.csv

COPY ValidationDataset.csv ValidationDataset.csv

ENTRYPOINT ["/opt/spark/bin/spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:2.7.7", "code_wine_predict_docker_1.py"]