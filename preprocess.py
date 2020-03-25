from utils import Logger, PascalVOCExtractor, AnnotationParser, IOProcessor
from config import Defaults

from PIL import Image
import numpy as np

import sys
from argparse import ArgumentParser
import os

# Get defaults options
opts = Defaults()

parser = ArgumentParser()
parser.add_argument(
    "--annotations_path", # double dash enables the default value
    default=opts.ANNOTATIONS_PATH,
    help="Path to the folder containing Pascal-VOC like XML annotations"
)
parser.add_argument(
    "--images_path",
    default=opts.IMAGES_PATH,
    help="Path to the folder containing JPEG images"
)
parser.add_argument(
    "--image_names",
    default=opts.IMAGES_NAMES,
    help="Path to the file containing the name of the images and annotations we want to visualize"
)
parser.add_argument(
    "--outputs_path",
    default=opts.OUTPUTS_PATH,
    help="Path to the folder containing the outputs of this script"
)
parser.add_argument(
    "--task",
    default=opts.TASK,
    help="Name of the task to perform: in ['classification', 'detection', 'segmentation']"
)
parser.add_argument(
    "--input_size",
    default=opts.INPUT_SIZE,
    help="Size of the desired input images"
)

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")
logger.log("Annotations path:", args.annotations_path)
logger.log("Images path:", args.images_path)
logger.log("Image names:", args.image_names)
logger.log("Output path:", args.outputs_path)
logger.log("Task:", args.task)
logger.log("Input Size:", args.input_size)

# Create output directory
os.makedirs(args.outputs_path, exist_ok=True)

# Extract name of the images 
names = PascalVOCExtractor().extract_names(args.image_names)
logger.log("Target images number:", len(names))

# Extract the full path of the images and of the annotations
im_paths, ann_paths = PascalVOCExtractor().extract_paths(args.images_path, args.annotations_path, names)

im_array, desc_array = IOProcessor().process(im_paths, ann_paths, args.input_size, args.task, opts.CLASSES)
logger.success("Processing successful")

# Save images (X) and description (y)
im_array_name = "images-array-{}-{}.npy".format(args.image_names.split("\\")[-1][:-4], args.input_size) # Platform specific: windows
desc_array_name = "desc-array-{}-{}.npy".format(args.image_names.split("\\")[-1][:-4], args.input_size) # Platform specific: windows
im_output = os.path.join(args.outputs_path, im_array_name)
desc_output = os.path.join(args.outputs_path, desc_array_name)

logger.log("Image array file name:", im_array_name)
logger.log("Image array path:", im_output)
logger.log("Description array file name:", desc_array_name)
logger.log("Description array path:", desc_output)


logger.log("Size of the images array to store:", "{0:.2f} MB".format(im_array.nbytes/1e6))
np.save(im_output, im_array)
logger.success("Storing successful")

logger.log("Size of the descriptions array to store:", "{0:.2f} KB".format(desc_array.nbytes/1e3))
np.save(desc_output, desc_array)
logger.success("Storing successful")