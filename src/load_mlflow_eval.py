import os

import argparse
import imageio.v2 as imageio
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import torch
import torchvision.transforms as transforms

from model import Net

# import pathlib
# import sys
# os.chdir(str(pathlib.Path(sys.argv[0]).parent.parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_uri',
        type=str,
        default='azureml://locations',
        help='Path to the training data'
    )
    args = parser.parse_args()

    # prepare 3x32x32 image
    print("Load the image.")
    image = imageio.imread(
        'https://cdn.sstatic.net/Sites/stackoverflow/img/logo.png')
    np_img = np.array(image)
    np_img = np.transpose(np_img, [2, 0, 1])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3]),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    pt_img = transform(np_img)
    pt_img = pt_img[None, :, :, :]

    print('Loading the model')
    # https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.save_model
    with mlflow.start_run() as run:
        net = mlflow.pytorch.load_model(args.model_uri)

    with torch.no_grad():
        outputs = net(pt_img)
        print(f'pred: {outputs}')
        signature = infer_signature(pt_img.numpy(), outputs.numpy())
        print(f'signature: {signature}')
        """Returns:
        inputs: 
        [Tensor('float32', (-1, 3, 32, 32))]
        outputs: 
        [Tensor('float32', (-1, 10))]
        """
