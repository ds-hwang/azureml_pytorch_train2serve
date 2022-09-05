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
        '--url',
        type=str,
        default='https://cdn.sstatic.net/Sites/stackoverflow/img/logo.png',
        help='Path to the training data'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='ckpt.pth',
        help='ckpt path'
    )
    parser.add_argument("--registered_model_name", type=str,
                        default='torch_model', help="model name")
    parser.add_argument("--model", type=str, default='mdl',
                        help="path to model file")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.pytorch.autolog()

    # prepare 3x32x32 image
    print("Load the image.")
    image = imageio.imread(args.url)
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

    # define convolutional network
    net = Net()
    print('Loading the model')
    net.load_state_dict(torch.load(args.ckpt))
    print('Eval the model')
    net.eval()

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

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    # https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.save_model
    # https://github.com/pytorch/pytorch/issues/18325#issuecomment-600457228
    scripted_pytorch_model = torch.jit.script(net)
    mlflow.pytorch.log_model(
        pytorch_model=scripted_pytorch_model,
        artifact_path=args.registered_model_name,
        registered_model_name=args.registered_model_name,
        signature=signature,
    )

    # Saving the model to a file
    os.makedirs(args.model, exist_ok=True)
    mlflow.pytorch.save_model(
        pytorch_model=scripted_pytorch_model,
        path=os.path.join(args.model, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()
