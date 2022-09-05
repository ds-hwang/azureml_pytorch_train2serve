# run-pytorch-data.py

from azureml.core import Workspace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)
from azure.identity import DefaultAzureCredential

import pathlib
import os
import sys
import uuid
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

    registered_model_name = "pytorch_eval_model"
    # Let's pick the latest version of the model
    latest_model_version = max(
        [int(m.version)
         for m in ml_client.models.list(name=registered_model_name)]
    )
    # picking the model to deploy. Here we use the latest version of our registered model
    model = ml_client.models.get(
        name=registered_model_name, version=latest_model_version)
    print(
        f"Find the model: {registered_model_name}, v: {latest_model_version}")

    # Creating a unique name for the endpoint
    online_endpoint_name = "torch-endpoint-" + str(uuid.uuid4())[:8]

    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is an online endpoint",
        auth_mode="key",
    )

    endpoint = ml_client.begin_create_or_update(endpoint)
    print(
        f"Endpint {endpoint.name} provisioning state: {endpoint.provisioning_state}")

    # create an online deployment.
    # https://docs.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list
    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=online_endpoint_name,
        model=model,
        instance_type="Standard_E2s_v3",
        instance_count=1,
    )

    blue_deployment = ml_client.begin_create_or_update(blue_deployment)
    print(f"Deploy {blue_deployment} done.")
