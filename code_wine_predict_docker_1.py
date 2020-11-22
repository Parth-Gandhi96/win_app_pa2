import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# check weather correctly imported or not
# print(pyspark.__version__)
# df = spark.sql("select 'spark' as hello ")
# df.show()


def loadDataFile(fileName):
    df = (spark.read
          .format("csv")
          .options(header='true',delimiter=';')
          .load(fileName+".csv"))
    print(type(df))
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
    return df2
# printing the data we just read from file
# df.toPandas()
# printing the data we just read from file
# df.toPandas()

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
    return dataset
# checking all column has coverted to data type float
# dataset.dtypes
# Checking whether there is any column will numm value or not, printing its count
# from pyspark.sql.functions import isnull, when, count, col

# dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

# Assemble all the features with VectorAssembler

from pyspark.ml.feature import VectorAssembler
def transformData(dataset):
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
    transformed_data = assembler.transform(dataset)
    return transformed_data

from pyspark.ml.classification import RandomForestClassifier
def trainModal(training_data):
    rf = RandomForestClassifier(labelCol='quality', 
                                featuresCol='features',
                                maxDepth=15, maxBins=25,numTrees=40)
    # Fitting training model in current ML model
    model = rf.fit(training_data)
    return model 	


# Predeciting the class for testing data
def predictLabel(test_data,model):
    predictions = model.transform(test_data)
    return predictions


# calulating the accuracy from the predicted labels for test data
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
def calAccuracy(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol='quality', 
        predictionCol='prediction', 
        metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    return accuracy

# shorwing the data with its actual and predicted label
# predictions.toPandas()


def run():
    training_data = transformData(dataProcessing(loadDataFile("TrainingDataset")));
    test_data = transformData(dataProcessing(loadDataFile("ValidationDataset")));
    print("Training Data has number of rows: ", training_data.count())
    print("Test Data has number of rows: ", test_data.count())
    model = trainModal(training_data);
    predictedLabels = predictLabel(test_data,model);
    accuracy = calAccuracy(predictedLabels);
    print('Test Accuracy = ', (100*accuracy),'%')


run();