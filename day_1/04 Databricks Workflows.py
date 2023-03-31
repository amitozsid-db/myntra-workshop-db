# Databricks notebook source
import requests

# COMMAND ----------

url = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiUrl()
    .getOrElse(None)
)
token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
current_user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .tags()
    .apply("user")
)

create_job_api_url = url + "/api/2.1/jobs/create"

# COMMAND ----------

job_definition = {
    "name": f"{current_user}_workshop_workflow_demo",
    "webhook_notifications": {},
    "timeout_seconds": 0,
    "max_concurrent_runs": 1,
    "email_notifications": {},
    "tasks": [
        {
            "task_key": "data_cleansing",
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Repos/{current_user}/myntra-workshop-db/day_1/02 - Data Cleansing",
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
        },
        {
            "task_key": "model_training",
            "depends_on": [{"task_key": "data_cleansing"}],
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Repos/{current_user}/myntra-workshop-db/day_1/03 - Linear Regression II",
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
        },
    ],
    "job_clusters": [
        {
            "job_cluster_key": "Job_cluster",
            "new_cluster": {
                "cluster_name": "",
                "spark_version": "11.3.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.master": "local[*, 4]",
                    "spark.databricks.cluster.profile": "singleNode",
                },
                "aws_attributes": {
                    "first_on_demand": 1,
                    "availability": "SPOT_WITH_FALLBACK",
                    "zone_id": "us-west-2c",
                    "spot_bid_price_percent": 100,
                    "ebs_volume_count": 0,
                },
                "node_type_id": "Standard_DS3_v2",
                "driver_node_type_id": "iStandard_DS3_v2",
                "custom_tags": {"ResourceClass": "SingleNode"},
                "enable_elastic_disk": "true",
                "data_security_mode": "SINGLE_USER",
                "runtime_engine": "STANDARD",
                "num_workers": 0,
            },
        }
    ],
    "format": "MULTI_TASK",
}

# COMMAND ----------

def create_job(create_job_api_url: str, job_definition: dict, token: str)->dict:
    """ """
    try:
        response = requests.post(
            url=create_job_api_url,
            headers={"Authorization": "Bearer %s" % token},
            json=job_definition,
        )
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    except Exception as e:
      #handle error
        print("Error")

# COMMAND ----------

print(create_job(create_job_api_url, job_definition, token))
