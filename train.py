from models import create_logistic_model, create_resnet50
from config import Defaults
from utils import load_data, Logger

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import os
import datetime
from argparse import ArgumentParser


# Get defaults options
opts = Defaults()

parser = ArgumentParser()

parser.add_argument(
    "--h5_path",
    help="Path to features and labels h5 file. Keys: {'features', 'labels'}",
    default=r"C:\Users\Atlas\Documents\MS Azure\pascal-voc\outputs\voc_classification_trainval_224.h5"
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
logger.log("Labels & features (.h5) path:", args.h5_path)
logger.log("Outputs path:", args.outputs_path)
logger.log("Logs path:", args.logs_path)

# Create output directory
os.makedirs(args.outputs_path, exist_ok=True)

# Load data
X, y = load_data(args.h5_path)
logger.success("Data loaded")

input_shape = X.shape[1:] # e.g. 416x416x3
num_classes = y.shape[-1] # e.g. 20

logger.log("Input shape:", input_shape)
logger.log("Number of classes:", num_classes)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
y_train = y_train.astype("float32")
y_val = y_val.astype("float32")

logger.success("Data splitted")
logger.log("X_train shape:", X_train.shape)
logger.log("X_val shape:", X_val.shape)
logger.log("y_train shape:", y_train.shape)
logger.log("y_val shape:", y_val.shape)

# Create model and compile it for training
# model = create_logistic_model(
#     input_shape, 
#     num_classes
# )
model = create_resnet50(
     input_shape,
     num_classes
)
logger.success("Model created")
model.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=["binary_accuracy", tf.metrics.Precision()]
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
    epochs=1,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)

# Save model and weights separately
model_path = os.path.join(
    args.outputs_path,
    "{}_{}_{}.json".format(model.name, input_shape[0], num_classes)
)
with open(model_path, "w") as file:
    file.write(model.to_json())
logger.success("Architecture saved in {}".format(model_path))

weights_path = os.path.join(
    args.outputs_path,
    "{}_{}_{}.h5".format(model.name, input_shape[0], num_classes)
)
model.save_weights(weights_path)
logger.success("Weights saved in {}".format(weights_path))

logger.log("Model and weights saved in:", args.outputs_path)