# Apache Spark Machine Learning with MLlib and Linear Regression on Databricks

## Introduction

Welcome to the Apache Spark Machine Learning project using MLlib and Linear Regression on Databricks! This project demonstrates the application of machine learning techniques on big data using PySpark, the Python API for Apache Spark. This guide will walk you through the entire process, from setting up your Databricks environment to performing data analysis and building a linear regression model.

## What is Apache Spark and PySpark?

### Apache Spark

Apache Spark is an open-source, distributed computing system designed for fast and efficient big data processing. It provides an interface for programming entire clusters with implicit data parallelism and fault tolerance.

### PySpark

PySpark is the Python API for Apache Spark. It allows Python developers to utilize the powerful distributed computing capabilities of Spark while writing code in Python, a more user-friendly language.

### Differences Between Apache Spark, PySpark, and Pandas

- **Apache Spark**: Best suited for large-scale data processing and analytics across clusters.
- **PySpark**: Provides the power of Apache Spark with the simplicity of Python.
- **Pandas**: A data manipulation library ideal for smaller datasets that can be handled on a single machine.

## Packages Used in This Project

- **pyspark**: Python API for Apache Spark, used for data processing and machine learning.
- **pandas**: Data manipulation library for data transformation and analysis on smaller datasets.
- **matplotlib**: Visualization library for creating static, animated, and interactive plots in Python.
- **seaborn**: Statistical data visualization library based on matplotlib.

## Project Setup

### Step 1: Create a Databricks Community Edition Account

1. Visit the [Databricks Community Edition](https://community.cloud.databricks.com/login.html) website.
2. Click on "Get Started for Free".
3. Fill in your details to create an account.
4. Verify your email address and log in to Databricks.

### Step 2: Create a New Cluster

1. After logging in, click on "Clusters" in the left-hand menu.
2. Click "Create Cluster".
3. Name your cluster (e.g., "Spark-ML-Cluster").
4. Select the appropriate Databricks runtime version.
5. Click "Create Cluster".

### Step 3: Upload Data to Databricks

You can either upload the data files directly to Databricks or use S3 for storage.

#### Option A: Upload Data Directly to Databricks

1. Click on "Data" in the left-hand menu.
2. Click "Add Data" and select "Upload File".
3. Upload the CSV files containing your data.

#### Option B: Use Amazon S3

1. If you have your data stored in S3, you can access it directly from Databricks.
2. Ensure you have the necessary AWS credentials configured.
3. Use the following code snippet to read data from S3:
   ```python
   df = spark.read.csv("s3a://your-bucket-name/your-file.csv", header=True, inferSchema=True)
   ```

### Step 4: Clone the Project Repository

1. In your Databricks workspace, click on "Repos" in the left-hand menu.
2. Click "Add Repo" and select "Clone Existing Repo".
3. Enter the URL of the repository: `https://github.com/TravelXML/APACHE-SPARK-PYSPARK-DATABRICKS-MACHINE-LEARNING-MLIB`
4. Click "Create Repo".

### Step 5: Open the Notebooks

1. Navigate to the cloned repository in the "Repos" section.
2. Open the notebook files `PYSPARK - LINER REGRESSION.ipynb` and `PYSPARK ML.ipynb`.

## Running the Analysis

### Step 1: Import Necessary Libraries

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

### Step 2: Create a Spark Session

```python
spark = SparkSession.builder.appName('Spark ML Example').getOrCreate()
```

### Step 3: Load and Prepare Data

Replace `'s3a://your-bucket-name/your-file.csv'` with the actual path to your data file.

```python
# Load the data
file_path = '/FileStore/shared_uploads/astartupcto@gmail.com/test1.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Index categorical columns
indexer = StringIndexer(inputCols=["sex", "smoker", "day", "time"],
                        outputCols=["sex_indexed", "smoker_indexed", "day_indexed", "time_index"])
df_r = indexer.fit(df).transform(df)

# Assemble features into a vector
featureassembler = VectorAssembler(inputCols=['tip', 'size', 'sex_indexed', 'smoker_indexed', 'day_indexed', 'time_index'],
                                   outputCol="Independent Features")
finalized_data = featureassembler.transform(df_r)

# Select relevant columns
finalized_data = finalized_data.select("Independent Features", "total_bill")
```

### Step 4: Split Data into Training and Testing Sets

```python
train_data, test_data = finalized_data.randomSplit([0.75, 0.25])
```

### Step 5: Train the Linear Regression Model

```python
regressor = LinearRegression(featuresCol='Independent Features', labelCol='total_bill')
regressor = regressor.fit(train_data)
```

### Step 6: Evaluate the Model

```python
# Make predictions
predictions = regressor.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="total_bill", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(labelCol="total_bill", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(labelCol="total_bill", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)

# Show predictions
predictions.select("Independent Features", "total_bill", "prediction").show()

# Print performance metrics
print(f"R²: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
```


## Conclusion

Congratulations! You have successfully set up a Databricks environment, uploaded data, and performed machine learning analysis using PySpark. You have learned how to preprocess data, build a linear regression model, and evaluate its performance.

For more in-depth tutorials and articles on Apache Spark, PySpark, and big data analytics, subscribe to our updates.

## Additional Resources

- [Databricks Documentation](https://docs.databricks.com/)
- [Apache Spark Documentation](https://spark.apache.org/documentation.html)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

Feel free to reach out if you have any questions or need further assistance. Happy coding!
