import sys
from argparse import ArgumentParser
import os
from utils import Logger, annotate_image
import matplotlib.pyplot as plt

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

# Create full path to images
images = list(map(lambda x: os.path.join(args.images_path, "{}.jpg".format(x.strip())), targets))
logger.log("One example of complete image path:", images[0])


# Create full path to annotations
annotations = list(map(lambda x: os.path.join(args.annotations_path, "{}.xml".format(x.strip())), targets))
logger.log("One example of complete annotation path:", annotations[0])


for img, annotation, target in list(zip(images, annotations, targets)):  
    im, desc = annotate_image(img, annotation)
    print(Logger.set_color("= {} =".format(target.strip()), "yellow"))
    for seq in desc:
        print(" ".join(list(map(lambda x: str(x), seq))))
    plt.imshow(im)
    plt.show()


