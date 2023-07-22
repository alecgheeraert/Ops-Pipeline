from azureml.core import Model
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

from utility import connection_azure

import dotenv
import os

dotenv.load_dotenv()
emotions = os.environ.get('emotions').split(',')
model_name = os.environ.get('model_name')

def prepare_environment(ws):
    env_name = os.environ.get('deploy_env')
    dependencies = os.environ.get('deploy_dep')

    env = Environment.from_conda_specification(env_name, file_path=dependencies)
    env.register(workspace=ws)

    return env

def prepare_deployment(ws, environment):
    service_name = os.environ.get('deploy_svc')
    entry_script = os.path.join('scripts', 'score.py')

    inference_config = InferenceConfig(entry_script=entry_script, environment=environment)
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    model = Model(ws, model_name)

    service = Model.deploy(workspace=ws,
                        name=service_name,
                        models=[model],
                        inference_config=inference_config,
                        deployment_config=aci_config,
                        overwrite=True)
    return service

def download(ws):
    local_model_path = os.environ.get('local_path')
    model = Model(ws, name=model_name)
    model.download(local_model_path, exist_ok=True)


if __name__ == '__main__':
    ws = connection_azure()

    deployment_azure = os.environ.get('deployment_azure')
    if deployment_azure == 'true':
        environment = prepare_environment(ws)
        service = prepare_deployment(ws, environment)
        service.wait_for_deployment(show_output=False)

    download(ws)