import os
os.system('apt update')
os.system('apt-get install openjdk-8-jdk -qq > /dev/null')
os.system('wget -q https://dlcdn.apache.org/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz')
os.system('tar xf spark-3.3.2-bin-hadoop3.tgz')
os.system('pip install -q findspark')
os.system('pip install pyspark')

#!sudo apt update
#!apt-get install openjdk-8-jdk-headless -qq > /dev/null
#!wget -q https://dlcdn.apache.org/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz

#!tar xf spark-3.3.1-bin-hadoop3.tgz

#!pip install -q findspark
#!pip install pyspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.3.2-bin-hadoop3"

from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import SparkFiles

spark = SparkSession \
       .builder \
       .appName("Hackathon") \
       .getOrCreate()

spark
