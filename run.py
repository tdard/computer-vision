import tensorflow as tf
from tensorflow.keras.models import model_from_json
from argparse import ArgumentParser
from config import Defaults
from utils import Logger, IOProcessor, draw_text
from PIL import Image
import numpy as np
import os
import json

opts = Defaults()

parser = ArgumentParser()

parser.add_argument(
    "--arch_path",
    default="./data/outputs/logistic_regressor_416_20.json", 
    help="Path to model architecture (.json)"
)
parser.add_argument(
    "--weights_path",
    default="./data/outputs/logistic_regressor_416_20.h5",
    help="Path to model weights (.h5)"
)
parser.add_argument(
    "--image_path",
    default=os.path.join(
        r"C:\Users\Atlas\Documents\MS Azure\pascal-voc\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages", 
        "2008_000002.jpg"
    ),
    help="Path to the image (.jpg) that we want to predict"
)

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")
logger.log("Architecture path:", args.arch_path)
logger.log("Weights path:", args.weights_path)
logger.log("Image path:", args.image_path)

with open(args.arch_path, "r") as architecture:
    model = model_from_json(architecture.read())

model.load_weights(args.weights_path)
print(model.summary())

image_path = args.image_path
input_size = model.input_shape[1]
seguir = True
while seguir:
    try:
        # Load and process image
        im_array = IOProcessor().process_image(image_path, input_size)
        # Do the freaking prediction
        pred = model.predict(im_array)[0]
        # Filter by threshold
        pred = np.where(pred >= 0.5, 1, 0)  # the threshold can be in argument
        # Convert binary values to classes
        classes = []
        for i, p in enumerate(pred):
            if p==1: classes.append(list(opts.CLASSES.keys())[i])
        # Add classification information on the image and display it
        im = Image.open(image_path).convert("RGBA")
        draw_text(im, classes, show=True)
    except FileNotFoundError:
        logger.alert("File not found. The provided image path is not correct, please try a new one")
    # Ask for input again!
    logger.alert("Provide a new .jpg image path, or say 'quit' to... quit")
    image_path = input()
    if image_path.strip().lower() == "quit":
        seguir = False
