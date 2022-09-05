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
    web_img = 'https://cdn.sstatic.net/Sites/stackoverflow/img/logo.png'
    my_job_inputs = {
        "url": Input(type=AssetTypes.URI_FILE, path=web_img),
        "ckpt": Input(type=AssetTypes.URI_FILE, path="ckpt.pth"),
        "registered_model_name": registered_model_name,
    }
    # https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-core-syntax#azure-ml-data-reference-uri
    # azureml_output = 'azureml://datastores/workspaceblobstore/paths/outputs/pred.txt'
    my_job_outputs = dict(
        model=Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")
    )

    # target name of compute where job will be executed
    computeName = "cpu-cluster"
    job = command(
        code="./src",
        # the parameter will match the training script argument name
        # inputs.url key should match the dictionary key
        command="""python save_mlflow_eval.py --url ${{inputs.url}} --ckpt ${{inputs.ckpt}} \
            --registered_model_name ${{inputs.registered_model_name}} --model ${{outputs.model}}
            """,
        inputs=my_job_inputs,
        outputs=my_job_outputs,
        environment=f"{env_name}@latest",
        compute=computeName,
        display_name="pytorch-eval",
    )

    returned_job = ml_client.create_or_update(job)
    aml_url = returned_job.studio_url
    print("Monitor your job at", aml_url)
