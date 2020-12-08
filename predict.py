# import findspark
# findspark.init()
import sys

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

conf = (SparkConf().setAppName("Predict wine app"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

if len(sys.argv) == 2:
    testFile = sys.argv[1]
print("==== DataSet is being Read ====")
#print(testFile)

# Read trained model
model = RandomForestModel.load(sc, "s3://mywineproject/sg2277-wqModel")
print(model)

# Reading TestValidation.csv
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://mywineproject/ValidationDataset.csv')

outputRdd = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(outputRdd.map(lambda x: x.features))
labelsAndPredictions = outputRdd.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)

# Overall Statistics

f1Score = metrics.fMeasure()
print("\n\n==== Summary Statatistics ====")
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted precision = %s" % metrics.weightedPrecision)
