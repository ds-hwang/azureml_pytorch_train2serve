# Train and server a pytorch model via Azure ML

## Serve a pytorch model via endpoint (i.e. realtime server) in Azure ML

This project consists of

- Train a pytorch model locally and save a checkpoint in local folder. (You can train by Azure compute cluster of course. Check the Azure tutorials.)
  - `python ./src/train.py`
- Load the ckpt to eval the model, and register it to Azure ML model via `mlflow`.
  - `python ./azure_1_save_mlflow.py`
- Create a endpoint (Azure real-time serving) and deploy the mlflow pytorch model to the Azure ML endpoint.
  - `python ./azure_2_serve_endpoint.py`
- Request an inference to the endpoint. You will use this for your application.
  - `python ./azure_3_invoke_endpoint.py`

### Change config.json for your subscription

You can deploy the simple pytorch model as mentions above. But one acition required is to change `config.json` according to your Azure ML subscription. [Tutorial 1/2/3](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world) will give idea what I'm talking. There is some free quota. I didn't pay any money for this project.

# Several pain points on Azure ML

- Azure ML documents are not well organized. All info spreads here and there. I had to read lots of documents until I feel it's enough. All YouTube video are not very helpful. Hope Microsoft to hire better contents producers.
  - There are good starting points; [Tutorial 1/2/3](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world), [Tutorial: Create production ML pipelines with Python SDK v2](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk)
- [The serving tutorial](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk) use `sklearn` (really?). I couldn't find any documents using `pytorch` or `tensorflow`. I need to test lots of my hypothesis via time consuming deployment trials.
- Azure ML deploys the model via `mlflow` in very sneaky way. It's not well documented. It took 2 days to figure out. There are zero sample code using `pytorch`, `mlflow` and Azure all together.
  - Unfortunately `mlflow` doesn't support `jax`. :(
