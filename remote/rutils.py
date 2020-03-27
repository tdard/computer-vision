from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

import shutil
import os

def copy_files(files, dest):
    """
    copy the files to the destination
    """
    # Create destination directory if not existing already
    os.makedirs(
        name=dest,
        exist_ok=True
    )
    # Copy files
    for file in files:
        try:
            shutil.copy(
                src=file,
                dst=dest
            )
        except: # I don't care about the error
            return False
    return True

def create_conda_environment(workspace, name, conda_dependencies, pip_dependencies):
    """
    Create an environment or retrieve it by its name from workspace
    Pip installs Python packages whereas conda installs packages which may contain software written in any language.
    e.g. TensorFlow, Scikit-Learn -> Conda, Matplotlib -> pip   
    """
    if name in Environment.list(workspace):
        env = Environment.get(
            workspace=workspace,
            name=name
        )
        print("The environment '{}' already existed for the workspace".format(name))
    else:
        env = Environment(name=name)
        env.docker.enabled = True
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages=conda_dependencies,
            pip_packages=pip_dependencies,
        )
        env.register(workspace=workspace)
    return env

def create_gpu_target(workspace, name):
    try:
        target = ComputeTarget(
            workspace=workspace,
            name=name
        )
        print("Found existing compute target, use it.")
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="Standard_NC12",
            max_nodes=8)

        target = ComputeTarget.create(
            workspace=workspace,
            name=name,
            provisioning_configuration=compute_config
        )
    target.wait_for_completion(show_output=True)
    return target

def get_available_vm_sizes_from_config():
    ws = Workspace.from_config()
    vm_sizes = AmlCompute.supported_vmsizes(workspace=ws, location="francecentral")
    return vm_sizes

def recursive_upload(datastore, source):
    for (dirpath, dirnames, filenames) in os.walk(source):
        #print(dirpath) # path in which we are. it starts at source
        #print(dirnames) # directories that we have in dirpath
        #print(filenames) # only the names of the files that we have in dirpath
        datastore.upload(
            src_dir=dirpath, 
            target_path=dirpath,
            overwrite=False,
            show_progress=True)
