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
im_output = os.path.join(args.output_path, im_array_name)

logger.log("Image array file name:", im_array_name)
logger.log("Image array path:", im_output)

logger.log("Size (in bytes) of the array to store:", im_array.nbytes)
np.save(im_output, im_array)


# Save description!
# For classification we do not bother complex description, only names are important!
classes = {
    "aeroplane" : 1,
    "bicycle" : 2,
    "bird" : 3,
    "boat" : 4,
    "bottle" : 5,
    "bus" : 6,
    "car" : 7,
    "cat" : 8,
    "chair" : 9, 
    "cow" : 10,
    "diningtable" : 11,
    "dog" : 12,
    "horse" : 13,
    "motorbike" : 14,
    "person" : 15,
    "pottedplant" : 16,
    "sheep" : 17,
    "sofa" : 18,
    "train" : 19,
    "tvmonitor" : 20
}
desc_array = np.zeros((data_count, 20), dtype="int8")
for k in range(data_count):
    desc = AnnotationParser().parse(ann_paths[k], "classification")
    one_hot = np.zeros((1, 20))
    for c in desc:
        one_hot[:, classes[c]-1] = 1 # Since multiple classes can be here it's not really one hot but anyway
    desc_array[k, :] = one_hot


desc_array_name = "desc-array-{}-{}.npy".format(args.image_names.split("\\")[-1][:-4], args.input_size) # Platform specific: windows
desc_output = os.path.join(args.output_path, desc_array_name)

logger.log("Description array file name:", desc_array_name)
logger.log("Description array path:", desc_output)

# Save the description
logger.log("Size (in bytes) of the array to store:", desc_array.nbytes)
np.save(desc_output, desc_array)

