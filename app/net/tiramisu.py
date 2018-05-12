from keras import models
from keras.layers import BatchNormalization, Activation, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Reshape, \
    Permute
from keras.optimizers import Adam
from keras.regularizers import l2

from app.net.jaccard_metrics import jaccard_coef, jaccard_coef_int


class Tiramisu(object):


    def __init__(self):
        self.model = self.getTiramisu()


    def getTiramisu(self):
        model = self.model = models.Sequential()
        # cropping
        # model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))

        model.add(Conv2D(48, kernel_size=(3, 3), padding='same',
                         input_shape=(224, 224, 3),
                         kernel_initializer="he_uniform",
                         kernel_regularizer=l2(0.0001)))
        # (5 * 4)* 2 + 5 + 5 + 1 + 1 +1
        # growth_m = 4 * 12
        # previous_m = 48
        self.__get_dense_block(5, 108)  # 5*12 = 60 + 48 = 108
        self.__get_transition_down(108)
        # self.__get_dense_block(5, 168)  # 5*12 = 60 + 108 = 168
        # self.__get_transition_down(168)
        # self.__get_dense_block(5, 228)  # 5*12 = 60 + 168 = 228
        # self.__get_transition_down(228)
        # self.__get_dense_block(5, 288)  # 5*12 = 60 + 228 = 288
        # self.__get_transition_down(288)
        # self.__get_dense_block(5, 348)  # 5*12 = 60 + 288 = 348
        # self.__get_transition_down(348)

        self.__get_dense_block(15, 108)  # m = 348 + 5*12 = 408

        # self.__get_transition_up(468, (468, 7, 7), (None, 468, 14, 14))  # m = 348 + 5x12 + 5x12 = 468.
        # self.__get_dense_block(5, 468)

        # self.__get_transition_up(348, (348, 14, 14), (None, 348, 28, 28))  # m = 288 + 5x12 + 5x12 = 408
        # self.__get_dense_block(5, 348)
        #
        # self.__get_transition_up(288, (288, 28, 28), (None, 288, 56, 56))  # m = 228 + 5x12 + 5x12 = 348
        # self.__get_dense_block(5, 348)

        # self.__get_transition_up(168, (168, 56, 56), (None, 168, 112, 112))  # m = 168 + 5x12 + 5x12 = 288
        # self.__get_dense_block(5, 288)

        self.__get_transition_up(108, (108, 112, 112), (None, 108, 224, 224))  # m = 108 + 5x12 + 5x12 = 228
        self.__get_dense_block(5, 228)

        model.add(Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer="he_uniform"))

        # model.add(Reshape((12, 224 * 224)))
        # model.add(Permute((2, 1)))
        # model.add(Activation('sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
        return model

    def __get_dense_block(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(BatchNormalization(axis=1))
            model.add(Activation('relu'))
            model.add(Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_initializer="he_uniform"))
            model.add(Dropout(0.2))

    def __get_transition_down(self, filters):
        model = self.model
        model.add(BatchNormalization(axis=1, gamma_regularizer=l2(0.0001), beta_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters, kernel_size=(1, 1), padding='same', kernel_initializer="he_uniform"))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    def __get_transition_up(self, filters, input_shape, output_shape):
        model = self.model
        model.add(Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=input_shape, kernel_initializer="he_uniform"))
