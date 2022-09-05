# run-pytorch-data.py

import codecs
import json
import numpy as np

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

    online_endpoint_name = 'torch-endpoint-6c2134e1'
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
    print(
        f'Endpint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
    )

    request_file = "./sample-request.json"
    if os.path.exists(request_file):
        os.remove(request_file)
    img = np.random.randint(255, size=[1, 3, 32, 32])
    request = {"input_data": img.tolist()}
    with open(request_file, "wb") as f:
        json.dump(request, codecs.getwriter('utf-8')(f))
    print(f'{request_file} is ready.')

    # test the blue deployment with some sample data
    debug = False
    response = ml_client.online_endpoints.invoke(
        endpoint_name=online_endpoint_name,
        request_file=request_file,
        deployment_name="blue",
        logging_enable=debug
    )
    print(f'CIFAR-10 logits: {response}')
