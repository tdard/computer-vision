from azureml.core import Workspace, Dataset
from azureml.data.datapath import DataPath

import os
from argparse import ArgumentParser
import tarfile


parser = ArgumentParser()
parser.add_argument("--subscription_id")
parser.add_argument("--resource_group")
parser.add_argument("--workspace_name")
parser.add_argument("--dest")

args = parser.parse_args()

# Get workspace
ws = Workspace(
    subscription_id=args.subscription_id,
    resource_group=args.resource_group,
    workspace_name=args.workspace_name
)

# Get dataset
voc_ds = Dataset.get_by_name(
    workspace=ws, 
    name='Pascal VOC 2012 Mirror Link'
)

# Create right directory
os.makedirs(
    name=args.dest,
    exist_ok=True
)

# Download dataset into this very directory
files = voc_ds.download(
    target_path=args.dest,
    overwrite=False
)
print(files)

# Extract this file
for file in files:
    # Extract content
    with tarfile.open(file) as fd:
        fd.extractall(args.dest)
    # Remove original tar file
    os.remove(file)
print(os.listdir(args.dest))


# get the datastore to upload prepared data
datastore = ws.get_default_datastore()

# upload the local file from src_dir to the target_path in datastore
datastore.upload(
    src_dir=args.dest, 
    target_path=args.dest, 
    overwrite=False,
    show_progress=True
)
print("Upload at {} complete".format(args.dest))

# Create file dataset!
paths = [DataPath(datastore=datastore, path_on_datastore=args.dest)]
file_dataset = Dataset.File.from_files(path=paths)

# Register it
ds_name = "Pascal VOC 2012"
file_dataset.register(
    workspace=ws, 
    name=ds_name,
    description="Pascal VOC 2012 train/validation data"
)
print("File dataset {} registered".format(ds_name))

# Around 3h to do that using Aml Compute !!!!!!