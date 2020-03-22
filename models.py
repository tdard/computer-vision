from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten


def create_logistic_model(input_shape, num_classes):
    model_in = Input(shape=input_shape)
    x = Flatten()(model_in)
    model_out = Dense(units=num_classes, activation='sigmoid')(x)
    model = Model(inputs=[model_in], outputs=model_out)
    return model





