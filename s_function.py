import os
import numpy as np
import tensorflow.keras.models as Kmodels
import tensorflow.keras.layers as Klayers
import tensorflow.keras.optimizers as Koptimizers


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable warning

# create actor model
def create_cnn_actor(num_action):
    input_layer = Klayers.Input(shape=(100,180,1)) # height * width
    cnn1 = Klayers.Conv2D(filters=24, kernel_size=(5,5), strides=(3, 3),
                            activation='relu', padding='same')(input_layer)
    pool1 = Klayers.MaxPooling2D(pool_size=(3, 3), padding='same')(cnn1)
    cnn2 = Klayers.Conv2D(filters=32,kernel_size=(4,4),strides=(2, 2),
                            activation='relu', padding='same')(pool1)
    pool2 = Klayers.MaxPooling2D(pool_size=(2, 2), padding='same')(cnn2)
    cnn3 = Klayers.Conv2D(filters=32,kernel_size=(2,2),strides=(1, 1),
                            activation='relu', padding='same')(pool2)
    flatten = Klayers.Flatten()(cnn3)
    dense1 = Klayers.Dense(256, activation='relu')(flatten)
    output_layer = Klayers.Dense(num_action, activation='softmax')(dense1)

    model = Kmodels.Model(inputs=input_layer, outputs=output_layer)

    return model


# create critic model
def create_cnn_critic(lr):
    model = Kmodels.Sequential()
    model.add(Klayers.Conv2D(filters=24,kernel_size=(5,5),strides=(3, 3),
                            input_shape=(100,180,1),activation='relu',
                            padding='same'))
    model.add(Klayers.MaxPooling2D(pool_size=(3, 3), padding='same'))
    model.add(Klayers.Conv2D(filters=32,kernel_size=(4,4),strides=(2, 2),
                            activation='relu', padding='same'))
    model.add(Klayers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Klayers.Conv2D(filters=32,kernel_size=(2,2),strides=(1, 1),
                            activation='relu', padding='same'))
    model.add(Klayers.Flatten())
    model.add(Klayers.Dense(256, activation='relu',))
    model.add(Klayers.Dense(1, activation=None,))

    model.compile(optimizer=Koptimizers.Adam(lr=lr), loss='mse')

    return model


# check folder
def chk_folder(folder_path):
    if not os.path.exists(os.path.dirname(folder_path)):
        try:
            os.makedirs(os.path.dirname(folder_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
