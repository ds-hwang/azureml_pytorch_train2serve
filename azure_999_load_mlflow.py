# run-pytorch-data.py

from azureml.core import Workspace
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input

import pathlib
import os
import sys
os.chdir(str(pathlib.Path(sys.argv[0]).parent))

if __name__ == "__main__":
    # get details of the current Azure ML workspace
    ws = Workspace.from_config()

    # default authentication flow for Azure applications
    default_azure_credential = DefaultAzureCredential()
    subscription_id = ws.subscription_id
    resource_group = ws.resource_group
    workspace = ws.name

    # client class to interact with Azure ML services and resources, e.g. workspaces, jobs, models and so on.
    ml_client = MLClient(
        default_azure_credential,
        subscription_id,
        resource_group,
        workspace)

    env_name = "pytorch-env"
    env_docker_image = Environment(
        image="pytorch/pytorch:latest",
        name=env_name,
        conda_file="pytorch-env.yml",
    )
    ml_client.environments.create_or_update(env_docker_image)

    # the key here should match the key passed to the command
    registered_model_name = "pytorch_eval_model"
    # https://ml.azure.com/runs/sweet_hook_4rznd64m1q?wsid=/subscriptions/e4008d0d-d6b9-4d84-b0d6-2e1cb9cfa8e8/resourceGroups/luxtella-rg/providers/Microsoft.MachineLearningServices/workspaces/test&tid=8a66840e-ffed-4d6e-aa42-dbf41ee9332c
    model_uri = "azureml://locations/eastus2/workspaces/4af6c6f7-ec2c-4cc2-8997-b543ace3c7ea/models/azureml_sweet_hook_4rznd64m1q_output_mlflow_log_model_-54946137/versions/1"
    my_job_inputs = {
        "model_uri": Input(type=AssetTypes.MLFLOW_MODEL, path=model_uri),
    }
    # target name of compute where job will be executed
    computeName = "cpu-cluster"
    job = command(
        code="./src",
        # the parameter will match the training script argument name
        # inputs.url key should match the dictionary key
        command="python load_mlflow_eval.py --model_uri ${{inputs.model_uri}}",
        inputs=my_job_inputs,
        environment=f"{env_name}@latest",
        compute=computeName,
        display_name="pytorch-eval",
    )

    returned_job = ml_client.create_or_update(job)
    aml_url = returned_job.studio_url
    print("Monitor your job at", aml_url)
