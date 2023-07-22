from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget
from azureml.core.compute import AmlCompute
from azureml.core.environment import Environment


import os

def connection_azure() -> Workspace:
    client_secret = os.environ.get('client_secret')
    client_id = os.environ.get('client_id')
    tenant_id = os.environ.get('tenant_id')

    workspace = os.environ.get('workspace')
    resource_group = os.environ.get('resource_group')
    subscription_id = os.environ.get('subscription_id')

    spa = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=client_id,
        service_principal_password=client_secret
    )

    return Workspace.get(
        name=workspace,
        auth=spa,
        subscription_id=subscription_id,
        resource_group=resource_group
    )

def prepare_compute(ws) -> ComputeTarget:
    compute_name = os.environ.get('compute_name')
    compute_min = os.environ.get('compute_min')
    compute_max = os.environ.get('compute_max')
    compute_sku = os.environ.get('compute_sku')

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print(compute_name)
            return compute_target
    else:
        provisioning = AmlCompute.provisioning_configuration(
            vm_size=compute_sku,
            min_nodes=compute_min,
            max_nodes=compute_max
        )

        compute_target = ComputeTarget.create(ws, compute_name, provisioning)
        compute_target.wait_for_completion(
            show_output=True,
            min_node_count=None,
            timeout_in_minutes=20
        )

        print(compute_target.get_status().serialize())

        return compute_target

def prepare_training_env(ws):
    env_name = os.environ.get('training_env')
    dependencies = os.environ.get('training_dep')
    
    env = Environment.from_conda_specification(env_name, file_path=dependencies)
    env.python.user_managed_dependencies = False
    env.register(workspace=ws)

    return env