import argparse
import datetime
import os

from azure.ai.ml import Input, MLClient, dsl, load_component
from azure.ai.ml.constants import AssetTypes, TimeZone
from azure.ai.ml.entities import (
    Data,
    Environment,
    JobSchedule,
    RecurrencePattern,
    RecurrenceTrigger,
)
from azure.identity import DefaultAzureCredential

# User
user_name = "david"
# env
parser = argparse.ArgumentParser()
parser.add_argument("--env", dest="env", help="environment", required=True)
args = parser.parse_args()
env = args.env
# Workspace
subscription_id = "59a62e46-b799-4da2-8314-f56ef5acf82b"
resource_group = "rg-azuremltraining"
workspace_name = "dummy-workspace"
compute_cluster_name = "aml-cluster"
# Directory
dependencies_dir = "dependencies"
data_prep_src_dir = "components/data_prep"
train_src_dir = "components/train"
# Pipeline run config
pipeline_dry_run = ["dev", "uat"]
pipeline_schedule = "prd"


# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)
print(f"Connected to {subscription_id}/{resource_group}/{workspace_name}")


# Create data asset
web_path = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

credit_data = Data(
    name=f"{env}_{user_name}_creditcard_defaults",
    version="1.0",
    description="Credit Card Data",
    tags={"creator": user_name},
    properties={"format": "CSV"},
    path=web_path,
    type=AssetTypes.URI_FILE,
)


# Register data asset
credit_data = ml_client.data.create_or_update(credit_data)
print(
    f"Dataset with name {credit_data.name} was registered to workspace,"
    "the dataset version is {credit_data.version}"
)


# Create job environment for pipeline
job_env_conda = Environment(
    name=f"{user_name}_environment",
    description="Custom environment for Credit Defaults Pipeline",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    tags={"creator": user_name},
    version="1.0",
)

# Register the environment
pipeline_job_env = ml_client.environments.create_or_update(job_env_conda)
print(
    f"Created environment {pipeline_job_env.name},"
    f"version {pipeline_job_env.version}"
)


# load component data_prep
data_prep_component = load_component(
    source=os.path.join(data_prep_src_dir, "data_prep.yaml")
)
# data_prep_component = ml_client.create_or_update(data_prep_component)
print(
    f"Component {data_prep_component.name},"
    f" with Version {data_prep_component.version} is registered"
)


# load component train
train_component = load_component(source=os.path.join(train_src_dir, "train.yaml"))
# train_component = ml_client.create_or_update(train_component)
print(
    f"Component {train_component.name},"
    f" with Version {train_component.version} is registered"
)


# Create pipeline
@dsl.pipeline(
    compute=compute_cluster_name,
    description=f"{user_name} E2E data_perp-train pipeline",
)
def credit_defaults_pipeline(
    pipeline_job_data_input,
    pipeline_job_test_train_ratio,
    pipeline_job_learning_rate,
    pipeline_job_registered_model_name,
):
    # using data_prep_function like a python call with its own inputs
    data_prep_job = data_prep_component(
        data=pipeline_job_data_input,
        test_train_ratio=pipeline_job_test_train_ratio,
    )
    # Add the data prep and training component
    train_component(
        train_data=data_prep_job.outputs.train_data,
        test_data=data_prep_job.outputs.test_data,
        learning_rate=pipeline_job_learning_rate,
        registered_model_name=pipeline_job_registered_model_name,
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data,
        "pipeline_job_test_data": data_prep_job.outputs.test_data,
    }


# Initialize the pipeline
registered_model_name = f"{env}_{user_name}_credit_defaults_model"
pipeline = credit_defaults_pipeline(
    pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
    pipeline_job_test_train_ratio=0.26,
    pipeline_job_learning_rate=0.0,
    pipeline_job_registered_model_name=registered_model_name,
)
print(f"Pipeline {pipeline.name} is created")


# Run the pipeline
if env in pipeline_dry_run:
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        # Project's name
        experiment_name=f"{env}_{user_name}_gha_credit",
    )
    print(f"runjob {pipeline_job.name} is started")
# Schedule the pipeline
if env == pipeline_schedule:
    schedule_name = f"{env}_{user_name}_credit"

    schedule_start_time = datetime.datetime.utcnow()
    recurrence_trigger = RecurrenceTrigger(
        frequency="month",
        interval=1,
        schedule=RecurrencePattern(month_days=1, hours=1, minutes=0),
        start_time=schedule_start_time,
        time_zone=TimeZone.ROMANCE_STANDARD_TIME,
    )

    job_schedule = JobSchedule(
        name=schedule_name, trigger=recurrence_trigger, create_job=pipeline
    )

    job_schedule = ml_client.schedules.begin_create_or_update(
        schedule=job_schedule
    ).result()
    print(job_schedule)
