import sys
from argparse import ArgumentParser
import os
from PIL import Image
from utils import Logger


# Defaults
cwd = os.getcwd()
# XML Folder containing annotations (detection / classification)
annotations_path = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations")
# Image Folder containing... images! (jpg for pascal voc)
images_path = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages")
# File containing the name of the images to visualize
image_names = os.path.join(cwd, r"data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main\trainval.txt")


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

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")
logger.log("Annotations path:", args.annotations_path)
logger.log("Images path:", args.images_path)
logger.log("Image names:", args.image_names)


# Retrieve targetted image names:
with open(args.image_names, "r") as file:
    targets = file.readlines()
logger.log("Target images number:", len(targets))

images = list(map(lambda x: os.path.join(args.images_path, "{}.jpg".format(x.strip())), targets))
logger.log("One example of complete image path:", images[0])

for path in images:
    img = Image.open(path, mode="r")
    img.show()
    img.close()
    if input(Logger.set_color("Press 'q' to quit\n", "red")) == "q":
        print("Stop image visualization")
        break
