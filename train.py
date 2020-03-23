from models import create_logistic_model
from config import Defaults
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_data, Logger
import datetime
from argparse import ArgumentParser


# Get defaults options
opts = Defaults()

parser = ArgumentParser()

parser.add_argument(
    "--images_array_path",
    default="./data/outputs/images-array-trainval-416.npy",
    help="Name of the .npy file containing images data"
)
parser.add_argument(
    "--descriptions_array_path",
    default="./data/outputs/desc-array-trainval-416.npy",
    help="Name of the .npy file containing descriptions data"
)
parser.add_argument(
    "--outputs_path",
    default=opts.OUTPUTS_PATH,
    help="Path to the outputs directory of the script"
)
parser.add_argument(
    "--logs_path",
    default=opts.LOGS_PATH,
    help="Path to the logs directory"
)

args = parser.parse_args()

Logger.enable_colors()
logger = Logger("blue")
logger.log("Images array (.npy) path:", args.images_array_path)
logger.log("Descriptions array (.npy) path:", args.descriptions_array_path)
logger.log("Outputs path:", args.outputs_path)
logger.log("Logs path:", args.logs_path)

# Create output directory
os.makedirs(args.outputs_path, exist_ok=True)

# Load data
images, descriptions = load_data(args.images_array_path, args.descriptions_array_path)
logger.success("Data loaded")

input_shape = images.shape[1:] # e.g. 416x416x3
num_classes = descriptions.shape[-1] # e.g. 20

logger.log("Input shape:", input_shape)
logger.log("Number of classes:", num_classes)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, descriptions, test_size=0.33, random_state=42)
logger.success("Data splitted")

# Create model and compile it for training
model = create_logistic_model(
    input_shape, 
    num_classes
)
logger.success("Model created")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
logger.success("Model compiled")
print(model.summary())

# Handle TensorBoard callbacks (logs)
os.makedirs(os.path.join(args.logs_path, "fit"), exist_ok=True)
logs_path = os.path.join(opts.LOGS_PATH, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs_path,
    histogram_freq=1
)

# Run training
model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=3,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)

# Save model and weights separately
model_path = os.path.join(
    args.outputs_path,
    "logistic_regressor_{}_{}.json".format(input_shape[0], num_classes)
)
with open(model_path, "w") as file:
    file.write(model.to_json())
logger.success("Architecture saved in {}".format(model_path))

weights_path = os.path.join(
    args.outputs_path,
    "logistic_regressor_{}_{}.h5".format(input_shape[0], num_classes)
)
model.save_weights(weights_path)
logger.success("Weights saved in {}".format(weights_path))

logger.log("Model and weights saved in:", args.outputs_path)