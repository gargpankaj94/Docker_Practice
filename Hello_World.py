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
