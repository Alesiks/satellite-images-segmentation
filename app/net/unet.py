from keras import Input, Model, layers
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.utils import plot_model

from app.config.main_config import IMAGE_SIZE
from app.net.jaccard_metrics import jaccard_coef, jaccard_coef_int


class UNET(object):

    def __init__(self):
        self.model = self.get_unet()

    def get_unet(self):
        inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

        conv1 = Convolution2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(inputs)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = layers.advanced_activations.ELU()(conv1)
        conv1 = Convolution2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(conv1)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(pool1)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = layers.advanced_activations.ELU()(conv2)
        conv2 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(conv2)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform')(pool2)
        conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = layers.advanced_activations.ELU()(conv3)
        conv3 = Convolution2D(128, (3, 3), activation='relu', padding="same", kernel_initializer='he_uniform')(conv3)
        conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(pool3)
        conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = layers.advanced_activations.ELU()(conv4)
        conv4 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(conv4)
        conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = layers.advanced_activations.ELU()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(pool4)
        conv5 = layers.advanced_activations.ELU()(conv5)
        conv5 = Convolution2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(conv5)
        conv5 = layers.advanced_activations.ELU()(conv5)
        conv5 = BatchNormalization(axis=1)(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(up6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = layers.advanced_activations.ELU()(conv6)
        conv6 = Convolution2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = layers.advanced_activations.ELU()(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(up7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = layers.advanced_activations.ELU()(conv7)
        conv7 = Convolution2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = layers.advanced_activations.ELU()(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(up8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = layers.advanced_activations.ELU()(conv8)
        conv8 = Convolution2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = layers.advanced_activations.ELU()(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Convolution2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(up9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = layers.advanced_activations.ELU()(conv9)
        conv9 = Convolution2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(conv9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = layers.advanced_activations.ELU()(conv9)

        conv10 = Convolution2D(1, (1, 1), activation='sigmoid', padding="same")(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model