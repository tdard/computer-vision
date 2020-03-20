import sys
from argparse import ArgumentParser
import os
from utils import Logger, PascalVOCExtractor, AnnotationParser
from PIL import Image
import numpy as np


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
# Task
task = "classification"


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
    default=task,
    help="Name of the task to perform: in ['classification', 'detection', 'segmentation']"
)
parser.add_argument(
    "--input_size",
    default=416,
    help="Size of the input images"
)

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")
logger.log("Annotations path:", args.annotations_path)
logger.log("Images path:", args.images_path)
logger.log("Image names:", args.image_names)
logger.log("Output path:", args.output_path)
logger.log("Task:", args.task)
logger.log("Input Size:", args.input_size)

# Create output directory
os.makedirs(args.output_path, exist_ok=True)

# Extract name of the images 
names = PascalVOCExtractor().extract_names(args.image_names)
data_count = len(names)
logger.log("Target images number:", data_count)

# Extract the full path of the images and of the annotations
im_paths, ann_paths = PascalVOCExtractor().extract_paths(args.images_path, args.annotations_path, names)
logger.log("One example of complete image path:", im_paths[0])
logger.log("One example of complete annotation path:", ann_paths[0])

# Ouvrir les chemins menant aux images pour en créer des images PIL et les sauver dans des arrays
# Parser les annotations et les sauver dans un array (nom: annotations_<task>.pkl?)

# Handle images
im_array = np.zeros((data_count, args.input_size, args.input_size, 3), dtype=int) # Channel last
for k in range(data_count):
    im = Image.open(im_paths[k])
    w, h = im.size
    # Reshape if necessary
    if max(im.size) > args.input_size:
        biggest_dim = max(im.size)
        ratio = biggest_dim/args.input_size
        w = int(w / ratio)
        h = int(h / ratio)
        im = im.resize(size=(w, h), resample=Image.BICUBIC)
    assert max(im.size) == args.input_size
    im = np.asarray(im) # converts also w,h,c -> h,w,c
    # Bottom and right zero-padding
    im_array[k, :h, :w, :] = im

# Save images!

# Handle description

# Save description!




