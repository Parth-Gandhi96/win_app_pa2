# s3://wineapp-parth/

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("S3CSVRead").getOrCreate()
# sc = spark.sparkContext 
# check weather correctly imported or not
# print(pyspark.__version__)
# df = spark.sql("select 'spark' as hello ")
# df.show()

def loadDataFile(fileName):
    df = (spark.read
          .format("csv")
          .options(header='true',delimiter=';')
          .load(fileName))
#     print(type(df))
    df2 = df.withColumnRenamed('"""""fixed acidity""""',"fixed acidity") \
        .withColumnRenamed('""""volatile acidity""""',"volatile acidity")\
        .withColumnRenamed('""""citric acid""""',"citric acid") \
        .withColumnRenamed('""""residual sugar""""',"residual sugar")\
        .withColumnRenamed('""""chlorides""""',"chlorides") \
        .withColumnRenamed('""""free sulfur dioxide""""',"free sulfur dioxide")\
        .withColumnRenamed('""""total sulfur dioxide""""',"total sulfur dioxide") \
        .withColumnRenamed('""""density""""',"density")\
        .withColumnRenamed('""""pH""""',"pH") \
        .withColumnRenamed('""""sulphates""""',"sulphates")\
        .withColumnRenamed('""""alcohol""""',"alcohol") \
        .withColumnRenamed('""""quality"""""',"quality")
#     df2.toPandas()
    return df2
# printing the data we just read from file

# df.dtypes
# Columns are not string, they ar enumeric values

from pyspark.sql.functions import col
def dataProcessing(df):
    dataset = df.select(col('fixed acidity').cast('float'),
                             col('volatile acidity').cast('float'),
                             col('citric acid').cast('float'),
                             col('residual sugar').cast('float'),
                             col('chlorides').cast('float'),
                             col('free sulfur dioxide').cast('float'),
                             col('total sulfur dioxide').cast('float'),
                             col('density').cast('float'),
                             col('pH').cast('float'),
                             col('sulphates').cast('float'),
                             col('alcohol').cast('float'),
                             col('quality').cast('float')
                            )
    dataset.show();
    return dataset
# checking all column has coverted to data type float
# dataset.dtypes
# Checking whether there is any column will numm value or not, printing its count
# from pyspark.sql.functions import isnull, when, count, col

# dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

# Assemble all the features with VectorAssembler

from pyspark.ml.feature import VectorAssembler
def getAssembler(dataset):
    required_features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid', 
                        'residual sugar', 
                        'chlorides',
                        'free sulfur dioxide', 
                        'total sulfur dioxide', 
                        'density', 
                        'pH',
                        'sulphates',
                        'alcohol'
                       ]
    print("Model considers total of ",len(required_features)," number of features.")

    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    return assembler

def transformData(dataset):
    assembler = getAssembler(dataset);
    transformed_data = assembler.transform(dataset)
    return transformed_data

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
def trainModal(training_data):
    rf = RandomForestClassifier(labelCol='quality', 
                                featuresCol='features',
                                numTrees=100, maxBins=484, maxDepth=25, minInstancesPerNode=5, seed=34)
    # Fitting training model in current ML model
    model = rf.fit(training_data)
#     assembler = getAssembler(training_data);
#     rfPipeline = Pipeline(stages=[assembler, rf])

#     fit = rfPipeline.fit(training_data)
    model.write().overwrite().save("s3://wineapp-parth/rf_model.model")
    return model

# from pyspark.ml.classification import LogisticRegression

# def trainModal(training_data):
#     lr = LogisticRegression(labelCol='quality', featuresCol='features',maxIter=10, regParam=0.3, elasticNetParam=0.8)
#     # Fit the model
#     lrModel = lr.fit(training_data)
#     return lrModel


# from pyspark.ml.classification import NaiveBayes
# def trainModal(training_data):
#     nb = NaiveBayes(labelCol='quality', featuresCol='features',smoothing=1.0, modelType="multinomial")
#     nbModel = nb.fit(training_data)
#     return nbModel

# from pyspark.ml import Pipeline
# from pyspark.ml.regression import DecisionTreeRegressor
# from pyspark.ml.feature import VectorIndexer
# from pyspark.ml.evaluation import RegressionEvaluator
# def trainModal(training_data):
#     featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)

from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import Pipeline
def load_model():
    rf = RandomForestClassificationModel.load("s3://wineapp-parth/rf_model.model/")
    return rf

# Predeciting the class for testing data
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
def predictLabel(test_data,rf):
#     assembler = getAssembler(test_data);
#     rfPipeline = Pipeline(stages=[assembler, rf])

#     fit = rfPipeline.fit(test_data)
    predictions = rf.transform(test_data) 

    return predictions

# calulating the accuracy from the predicted labels for test data
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
def calAccuracy(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol='quality', 
        predictionCol='prediction', 
        metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol='quality', 
        predictionCol='prediction', 
        metricName='f1')
    f1Score = evaluator_f1.evaluate(predictions)
    return accuracy,f1Score



# shorwing the data with its actual and predicted label
# predictions.toPandas()


def run(filePath):
    test_data = transformData(dataProcessing(loadDataFile(filePath)));
    print("\n\n\n Test Data has number of rows: ", test_data.count())
    rf_model = load_model();
    predictedLabels = predictLabel(test_data,rf_model);
#     test_data.show()
    accuracy,f1Score = calAccuracy(predictedLabels);
    print('Test Accuracy = ', (100*accuracy),'f1 score = ', (100*f1Score),'\n')



import argparse
parser = argparse.ArgumentParser(description='Wine Quality prediction')
parser.add_argument('--test_file', required=True, help='please provide test file path you can provide s3 path or local file path')
args = parser.parse_args()
print("Argument passsed: "+args.test_file)
run(args.test_file);