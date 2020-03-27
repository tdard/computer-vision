from rutils import create_gpu_target, copy_files, create_conda_environment

from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator

import shutil

# Get Workspace
ws = Workspace.from_config()
print("Workspace created")

# Create experiment
exp_name = "voc-download"
exp = Experiment(
    workspace=ws,
    name=exp_name
)
print("Experiment '{}' created".format(exp_name))

# Copy what we need
script_folder = "remote-download-dataset"
files = ["computer-vision/remote/download_dataset.py"]
res = copy_files(
    files=files,
    dest=script_folder
)
assert res == True
print("Necessary files copied")

# Create environment, the one for the project
env_name = "cv-tensorflow"
env = create_conda_environment(
    workspace=ws,
    name=env_name,
    conda_dependencies=["tensorflow==2.0", "scikit-learn"],
    pip_dependencies=["azureml-defaults", "matplotlib", "progressbar2", "Pillow"]
)
env.python.conda_dependencies.add_pip_package("azureml-defaults")
print("Conda environment '{}' created".format(env_name))

# Create compute target -> It will be Unix I assume
target_name = "gpu-nc12"
target = create_gpu_target(
    workspace=ws,
    name=target_name
)
print("Compute target '{}' created".format(env_name))

# Create estimator
script_params = {
    '--subscription_id' : ws.subscription_id,
    '--resource_group' : ws.resource_group,
    '--workspace_name' : ws.name,
    '--dest' : "data"
}
print("Script parameters:", script_params)

est = Estimator(
    source_directory=script_folder,
    entry_script="download_dataset.py",
    script_params=script_params,
    compute_target=target,
    environment_definition=env,
)

# Run experiment
run = exp.submit(est)
run.wait_for_completion(show_output=True)

# Remove folder after use (after all we only copy a few python scripts, not huge data)
print("Attempt to clean following folder:", script_folder)
shutil.rmtree(script_folder, ignore_errors=True)
