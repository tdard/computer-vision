import sys
from argparse import ArgumentParser
import os
from utils import Logger

# Defaults
cwd = os.getcwd()
# XML Folder containing annotations (detection / classification)
annotations_path = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations")
# Image Folder containing... images! (jpg for pascal voc)
images_path = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages")
# File containing the name of the images to visualize
image_names = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main\trainval.txt")
# Output folder
output_path = os.path.join(cwd, r"data\outputs")

parser = ArgumentParser()
parser.add_argument(
    "--annotations_path", # double dash enables the default value
    default=annotations_path,
    help="Path to the folder containing Pascal-VOC like XML annotations"
)
parser.add_argument(
    "--images_path",
    default=images_path,
    help="Path to the folder containing JPEG images"
)
parser.add_argument(
    "--image_names",
    default=image_names,
    help="Path to the file containing the name of the images and annotations we want to visualize"
)
parser.add_argument(
    "--output_path",
    default=output_path,
    help="Path to the folder containing the output of this script"
)
parser.add_argument(
    "--task",
    default="classification",
    help="Name of the task to perform: in ['classification', 'detection', 'segmentation']"
)

args = parser.parse_args()