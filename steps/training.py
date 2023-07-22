from azureml.core import Dataset
from azureml.core import ScriptRunConfig
from azureml.core import Experiment

from utility import connection_azure
from utility import prepare_compute
from utility import prepare_training_env

from typing import Tuple

import dotenv
import os

dotenv.load_dotenv()
emotions = os.environ.get('emotions').split(',')
model_name = os.environ.get('model_name')
learning_rate = os.environ.get('learning_rate')
max_epochs = os.environ.get('max_epochs')
batch_size = os.environ.get('batch_size')
patience = os.environ.get('patience')

def training(ws, compute_target, environment) -> Tuple[Experiment, ScriptRunConfig]:
    experiment_name = 'emotions-classification'
    folder = 'scripts'
    dataset_training = os.environ.get('dataset_training')
    dataset_testing = os.environ.get('dataset_testing')

    datasets = Dataset.get_all(workspace=ws)
    exp = Experiment(workspace=ws, name=experiment_name)

    args = [
        '--folder-training', datasets[dataset_training].as_mount('/data/train'),
        '--folder-testing', datasets[dataset_testing].as_mount('/data/test'),
        '--max-epochs', max_epochs,
        '--learning-rate-init', learning_rate,
        '--batch-size', batch_size,
        '--patience', patience,
        '--model-name', model_name
    ]

    config = ScriptRunConfig(
        source_directory=folder,
        script='train.py',
        arguments=args,
        compute_target=compute_target,
        environment=environment
    )

    return exp, config

def register(ws, run):
    model_path = 'outputs/' + model_name
    datasets = Dataset.get_all(workspace=ws)
    dataset_testing = os.environ.get('dataset_testing')

    run.download_files(prefix=model_path)
    run.register_model(
        model_name,
        model_path=model_path,
        tags={'ai-model': 'cnn'},
        description="Image classification on emotions",
        sample_input_dataset=datasets[dataset_testing]
    )

if __name__ == '__main__':
    ws = connection_azure()

    compute_target = prepare_compute(ws)
    environment = prepare_training_env(ws)

    exp, config = training(ws, compute_target, environment)
    run = exp.submit(config=config)
    run.wait_for_completion(show_output=False)
    print(run.id)

    register(ws, run)