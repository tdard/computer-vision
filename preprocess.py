import sys
from argparse import ArgumentParser
import os
from utils import Logger, PascalVOCExtractor, AnnotationParser
from config import Defaults
from PIL import Image
import numpy as np


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
    help="Size of the input images"
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
data_count = len(names)
logger.log("Target images number:", data_count)

# Extract the full path of the images and of the annotations
im_paths, ann_paths = PascalVOCExtractor().extract_paths(args.images_path, args.annotations_path, names)
logger.log("One example of complete image path:", im_paths[0])
logger.log("One example of complete annotation path:", ann_paths[0])

# Handle images
im_array = np.zeros((data_count, args.input_size, args.input_size, 3), dtype="float16") # Channel last
for k in range(data_count):
    im = Image.open(im_paths[k])
    w, h = im.size
    # Reshape if necessary
    if max(im.size) > args.input_size:
        ratio = max(im.size)/args.input_size
        w = int(w / ratio)
        h = int(h / ratio)
        im = im.resize(size=(w, h), resample=Image.BICUBIC)
    assert max(im.size) == args.input_size
    # Cast PIL image in numpy array and normalize features between [0,1]
    im = np.asarray(im)/255 # converts also w,h,c -> h,w,c
    # Bottom and right zero-padding
    im_array[k, :h, :w, :] = im
    break

# Save images!
im_array_name = "images-array-{}-{}.npy".format(args.image_names.split("\\")[-1][:-4], args.input_size) # Platform specific: windows
im_output = os.path.join(args.outputs_path, im_array_name)

logger.log("Image array file name:", im_array_name)
logger.log("Image array path:", im_output)

logger.log("Size (in bytes) of the array to store:", im_array.nbytes)
np.save(im_output, im_array)


# Save description!
# For classification we do not bother complex description, only names are important!
desc_array = np.zeros((data_count, 20), dtype="int8")
for k in range(data_count):
    desc = AnnotationParser().parse(ann_paths[k], "classification")
    one_hot = np.zeros((1, 20))
    for c in desc:
        one_hot[:, opts.CLASSES[c]-1] = 1 # Since multiple classes can be here it's not really one hot but anyway
    desc_array[k, :] = one_hot


desc_array_name = "desc-array-{}-{}.npy".format(args.image_names.split("\\")[-1][:-4], args.input_size) # Platform specific: windows
desc_output = os.path.join(args.outputs_path, desc_array_name)

logger.log("Description array file name:", desc_array_name)
logger.log("Description array path:", desc_output)

# Save the description
logger.log("Size (in bytes) of the array to store:", desc_array.nbytes)
np.save(desc_output, desc_array)

