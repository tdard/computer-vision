from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50


def create_logistic_model(input_shape, num_classes):
    model_in = Input(shape=input_shape)
    x = Flatten()(model_in)
    x = Dense(units=128, activation='sigmoid')(x)
    x = Dense(units=128, activation="sigmoid")(x)
    model_out = Dense(units=num_classes, activation='sigmoid')(x)
    model = Model(inputs=[model_in], outputs=model_out, name="logistic_regressor")
    return model

def create_resnet50(input_shape, num_classes):
    # Input
    model_in = Input(shape=input_shape)
    # Base ResNet
    base = ResNet50(weights='imagenet', include_top=True, input_tensor=model_in)
    base.layers.pop()
    # Make layers trainable
    for layer in base.layers:
        layer.trainable = True
    # Add last layer
    model_out = Dense(units=num_classes, activation="sigmoid", name="fine_tune_probs")(base.layers[-1].output)
    # Create model
    model = Model(inputs=[model_in], outputs=model_out, name="FineTunedResNet50")
    return model