import sys
from argparse import ArgumentParser
import os
from utils import Logger, annotate_image, PascalVOCExtractor
from config import Defaults
import matplotlib.pyplot as plt

# Defaults
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

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")
logger.log("Annotations path:", args.annotations_path)
logger.log("Images path:", args.images_path)
logger.log("Image names:", args.image_names)


# Extract name of the images 
names = PascalVOCExtractor().extract_names(args.image_names)
logger.log("Target images number:", len(names))

# Extract the full path of the images and of the annotations
images, annotations = PascalVOCExtractor().extract_paths(args.images_path, args.annotations_path, names)
logger.log("One example of complete image path:", images[0])
logger.log("One example of complete annotation path:", annotations[0])


for img, annotation, name in list(zip(images, annotations, names)):  
    im, desc = annotate_image(img, annotation)
    print(Logger.set_color("= {} =".format(name), "yellow"))
    for seq in desc:
        print(" ".join(list(map(lambda x: str(x), seq))))
    plt.imshow(im)
    plt.show()


