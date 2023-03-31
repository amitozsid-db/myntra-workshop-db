# Databricks notebook source
# MAGIC %md # Create a Custom MLFlow component

# COMMAND ----------

# MAGIC %md ## Train a basic RF model

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
x = iris.data
y = iris.target

#define a function that prints the iris' classification based on the algorithm's output
def classifyiris(z):
    if z[0] == 0:
        print("The iris is setosa.\n")
    elif z[0] == 1:
        print("The iris is versicolor.\n")
    else:
        print("The iris is virginica.\n")


# Predict Using Random Forest
random_forest = RandomForestClassifier()
model = random_forest.fit(x,y)
z = random_forest.predict([[3,5,4,2]])
print("Using the Random Forest Classification =", random_forest.predict([[3,5,4,2]]))
classifyiris(z)

print("Saving model to rf_model.sav")
pickle.dump(model, open("rf_model.sav", 'wb'))

# COMMAND ----------

# MAGIC %md ## Create a Custom MLflow Model

# COMMAND ----------

import mlflow.pyfunc
import sklearn
import pickle

# Create an `artifacts` dictionary that assigns a unique name to the saved XGBoost model file.
# This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
# into the new MLflow Model's directory.
artifacts = {
    "embedded_model": "rf_model.sav",
}

# Define the model class

class ModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, version):      
        self.version = version
        
    def load_context(self, context):
        import pickle
        self.embedded_model = pickle.load(open(context.artifacts['embedded_model'], 'rb'))

    def predict(self, context, model_input):
        predictions = self.embedded_model.predict(model_input)
        print(f"predicting {model_input}")
        return [{'version': self.version, 'prediction': p} for p in predictions]

# COMMAND ----------

# MAGIC %md ## Save the Custom Model

# COMMAND ----------

current_user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .tags()
    .apply("user")
)

# COMMAND ----------

# Create a Conda environment for the new MLflow Model that contains the XGBoost library
# as a dependency, as well as the required CloudPickle library
import cloudpickle
import sklearn
import mlflow
import mlflow.pyfunc

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'pip',
      {'pip': [
        'scikit-learn=={}'.format(sklearn.__version__),
        'cloudpickle=={}'.format(cloudpickle.__version__)
      ]
      }
    ],
    'name': 'hm_model_env'
}

mlflow.set_experiment(f"/Users/{current_user}/{current_user}-custom-model")

with mlflow.start_run(run_name="So that I have a model registered") as mlflow_run:
  # Save the MLflow Model
  mlflow_pyfunc_model_path = "model_wrapper_pyfunc"
  mlflow.pyfunc.log_model(
          artifact_path=mlflow_pyfunc_model_path, python_model=ModelWrapper(version="0.0.1"), artifacts=artifacts,
          conda_env=conda_env) 

# COMMAND ----------

# MAGIC %md ## Load back and predict
# MAGIC 
# MAGIC _Zoltan: I had to remove a cell from here so the cells below might break_

# COMMAND ----------

artifact_path = mlflow.search_runs(order_by=["attributes.start_time DESC"])['artifact_uri'].iloc[0]
print(artifact_path)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(artifact_path + "/" + mlflow_pyfunc_model_path)
loaded_model.predict([[3,5,4,2]])
