from rutils import copy_files, create_conda_environment, create_gpu_target

from azureml.core import Workspace, Experiment, Dataset
from azureml.train.dnn import TensorFlow

import shutil


# Get workspace
ws = Workspace.from_config()
print("Workspace created")

# Create experiment
exp_name = "voc-train"
exp = Experiment(
    workspace=ws,
    name=exp_name
)
print("Experiment '{}' created".format(exp_name))

# Copy what we need
script_folder = "remote-train"
files = [
    "computer-vision/train.py",
    "computer-vision/config.py",
    "computer-vision/utils.py",
    "computer-vision/models.py"
]
res = copy_files(
    files=files,
    dest=script_folder
)
assert res == True

# Create compute target -> It will be Unix I assume
target_name = "azuresome-gpu"
target = create_gpu_target(
    workspace=ws,
    name=target_name
)
print("Compute target '{}' created".format(target_name))

# Get dataset
voc = Dataset.get_by_name(
    workspace=ws,
    name="voc-classification"
)

# Create estimator
script_params = {
    "--h5_path" : voc.as_named_input("voc_classification").as_download(),
    "--outputs_path" : "classification",
    "--logs_path" : "logs"
}
print("Script parameters:", script_params)

est = TensorFlow(
    source_directory=script_folder,
    compute_target=target,
    entry_script="train.py",
    use_gpu=True,
    script_params=script_params,
    framework_version="2.0",
    conda_packages=["scikit-learn"],
    pip_packages=["azureml-defaults", "matplotlib", "progressbar2", "Pillow", "h5py"]
)

# Run experiment
try:
    run = exp.submit(est)
    run.wait_for_completion(show_output=True)
except Exception as e:
    print(e)
    print("Experiment failed")

# Remove folder after use (after all we only copy a few python scripts, not huge data)
print("Attempt to clean following folder:", script_folder)
shutil.rmtree(script_folder, ignore_errors=True)