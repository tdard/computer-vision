from rutils import copy_files, create_conda_environment, create_gpu_target

from azureml.core import Workspace, Experiment, Dataset
from azureml.train.estimator import Estimator

import shutil

# Get workspace
ws = Workspace.from_config()
print("Workspace created")

# Create experiment
exp_name = "voc-preprocess"
exp = Experiment(
    workspace=ws,
    name=exp_name
)
print("Experiment '{}' created".format(exp_name))

# Copy what we need
script_folder = "remote-preprocess"
files = [
    "computer-vision/preprocess.py",
    "computer-vision/config.py",
    "computer-vision/utils.py",
]
res = copy_files(
    files=files,
    dest=script_folder
)
assert res == True

# Create environment, the one for the project
env_name = "cv-tensorflow"
env = create_conda_environment(
    workspace=ws,
    name=env_name,
    conda_dependencies=["tensorflow==2.0", "scikit-learn"],
    pip_dependencies=["azureml-defaults", "matplotlib", "progressbar2", "Pillow"]
)
env.python.conda_dependencies.add_pip_package("azureml-defaults")
env.python.conda_dependencies.add_pip_package("h5py")
print("Conda environment '{}' created".format(env_name))

# Create compute target -> It will be Unix I assume
target_name = "awesome-cpu-vm"
target = create_gpu_target(
    workspace=ws,
    name=target_name
)
print("Compute target '{}' created".format(target_name))

# Get dataset
voc = Dataset.get_by_name(
    workspace=ws,
    name="Pascal VOC 2012"
)

# Create estimator
script_params = {
    "--subscription_id" : ws.subscription_id,
    "--resource_group" : ws.resource_group,
    "--workspace_name" : ws.name,
    "--data_folder" : voc.as_named_input("voc").as_mount(),
    "--annotations_path" : "VOCdevkit/VOC2012/Annotations",
    "--images_path" : "VOCdevkit/VOC2012/JPEGImages",
    "--image_names" : "VOCdevkit/VOC2012/ImageSets/Main/trainval.txt",
    "--outputs_path" : "classification",
    "--task" : "classification",
    "--input_size" : 224
}
print("Script parameters:", script_params)

est = Estimator(
    source_directory=script_folder,
    entry_script="preprocess.py",
    script_params=script_params,
    compute_target=target,
    environment_definition=env,
)

# Run experiment
run = exp.submit(est)
run.wait_for_completion(show_output=True)

# Upload output files on datastore and dataset
script_params = {
    "--subscription_id" : ws.subscription_id,
    "--resource_group" : ws.resource_group,
    "--workspace_name" : ws.name,
    "--h5_path" : "classification/voc_classification_trainval_224.h5",
    "--outputs_path" : "classification"
}
print("Script parameters:", script_params)

est = Estimator(
    source_directory=script_folder,
    entry_script="upload_features_and_labels.py",
    script_params=script_params,
    compute_target=target,
    environment_definition=env,
)

run = exp.submit(est)
run.wait_for_completion(show_output=True)

# Remove folder after use (after all we only copy a few python scripts, not huge data)
print("Attempt to clean following folder:", script_folder)
shutil.rmtree(script_folder, ignore_errors=True)