from utils import Logger, PascalVOCExtractor, AnnotationParser, IOProcessor
from config import Defaults

from PIL import Image
import numpy as np
import h5py

import sys
from argparse import ArgumentParser
import os

# Get defaults options
opts = Defaults()

parser = ArgumentParser()
parser.add_argument(
    "--subscription_id"
)
parser.add_argument(
    "--resource_group"
)
parser.add_argument(
    "--workspace_name"
)
parser.add_argument(
    "--data_folder",
    type=str,
    default=opts.DATA_FOLDER,
    help="Absolute path of the folder containing the dataset"
)
parser.add_argument(
    "--annotations_path",
    default=opts.ANNOTATIONS_PATH,
    help="Relative path to the folder containing Pascal-VOC like XML annotations"
)
parser.add_argument(
    "--images_path",
    default=opts.IMAGES_PATH,
    help="Relative path to the folder containing JPEG images"
)
parser.add_argument(
    "--image_names",
    default=opts.IMAGES_NAMES,
    help="Relative path to the file containing the name of the images and annotations we want to visualize"
)
parser.add_argument(
    "--outputs_path",
    default=opts.OUTPUTS_PATH,
    help="Relative path to the folder containing the outputs of this script"
)
parser.add_argument(
    "--task",
    default=opts.TASK,
    help="Name of the task to perform: in ['classification', 'detection', 'segmentation']"
)
parser.add_argument(
    "--input_size",
    type=int,
    default=opts.INPUT_SIZE,
    help="Size of the desired input images"
)

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")

logger.log("Data folder:", args.data_folder)
logger.log("Annotations path:", args.annotations_path)
logger.log("Images path:", args.images_path)
logger.log("Image names:", args.image_names)
logger.log("Output path:", args.outputs_path)
logger.log("Task:", args.task)
logger.log("Input Size:", args.input_size)

# Create output directory
os.makedirs(args.outputs_path, exist_ok=True)

# Extract name of the images 
names = PascalVOCExtractor().extract_names(
    image_names=os.path.join(args.data_folder, args.image_names)
)
logger.log("Target images number:", len(names))

# Extract the full path of the images and of the annotations
images_paths, annotations_paths = PascalVOCExtractor().extract_paths(
    os.path.join(args.data_folder, args.images_path),
    os.path.join(args.data_folder, args.annotations_path), 
    names
)

features, labels = IOProcessor().process(images_paths, annotations_paths, args.input_size, args.task, opts.CLASSES)
logger.success("Processing successful")

# Save everything in h5!!
h5_name = "voc_classification_trainval_224.h5"
h5_path = os.path.join(args.outputs_path, h5_name)
with h5py.File(h5_path, "w") as file:
    file.create_dataset("features", data=features, dtype='float32')
    file.create_dataset("labels", data=labels, dtype='float32')
logger.success("Created h5 file {} successfully".format(h5_name))
logger.log("Absolute path of h5 file:", os.path.abspath(h5_path))


#### Try to do that on next experiment ####

# Update on datastore if it is not a windows platform
if not sys.platform.startswith("win"):
    logger.alert("This is not a windows platform, we try to update our features and labels on a datastore")

    from azureml.core import Workspace, Dataset
    from azureml.data.datapath import DataPath
    # Get workspace
    ws = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    files = [
        h5_path
    ]
    datastore = ws.get_default_datastore()
    datastore.upload_files(
        files=files,
        relative_root=args.outputs_path,
        target_path=args.outputs_path
    )
    logger.success("Files uploaded to '{}' in the datastore".format(args.outputs_path))

    # Create dataset and register it
    paths = [
        DataPath(datastore=datastore, path_on_datastore=h5_path),
    ]
    dataset = Dataset.File.from_files(path=paths)

    ds_name = "voc-classification"
    dataset.register(
        workspace=ws,
        name=ds_name,
        description="Preprocessed features and labels of Pascal VOC 2012: 0-padding, resizing and features normalization on classification task"
    )
    print("File dataset {} registered".format(ds_name))