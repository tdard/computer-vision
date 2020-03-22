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
    "--images_array",
    default="images-array-trainval-416.npy",
    help="Name of the .npy file containing images data"
)
parser.add_argument(
    "--descriptions_array",
    default="desc-array-trainval-416.npy",
    help="Name of the .npy file containing descriptions data"
)
parser.add_argument(
    "--inputs_path",
    default=opts.OUTPUTS_PATH,
    help="Path to the directory where images_array and descriptions_array are stored"
)
parser.add_argument(
    "--outputs_path",
    default=opts.OUTPUTS_PATH,
    help="Path to the outputs directory of the script"
)

args = parser.parse_args()
logger = Logger("blue")

images, descriptions = load_data(args.inputs_path, args.images_array, args.descriptions_array)

input_shape = images.shape[1:] # e.g. 416x416x3
num_classes = descriptions.shape[-1] # e.g. 20

logger.log("Input shape:", input_shape)
logger.log("Number of classes:", num_classes)

X_train, X_val, y_train, y_val = train_test_split(images, descriptions, test_size=0.33, random_state=42)

model = create_logistic_model(
    input_shape, 
    num_classes
)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

os.makedirs("logs/fit/", exist_ok=True)
log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=3,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)

# Save model and weights separately
model_file_name = "logistic_regressor_{}_{}.json".format(input_shape[0], num_classes)
with open(os.path.join(args.outputs_path, model_file_name), "w") as file:
    file.write(model.to_json())

weights_file_name = "logistic_regressor_{}_{}.h5".format(input_shape[0], num_classes)
model.save_weights(os.path.join(args.outputs_path, weights_file_name))

logger.log("Model and weights saved in:", args.outputs_path)