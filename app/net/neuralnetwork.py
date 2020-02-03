import os

import tifffile as tiff
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


from app.config.main_config import TRAIN_INPUT_DATA_PATH, TRAIN_OUTPUT_DATA_PATH, VALIDATION_INPUT_DATA_PATH, \
    VALIDATION_OUTPUT_DATA_PATH, IMAGE_FORMAT, IMAGE_SIZE
from app.net.jaccard_metrics import jaccard_coef, jaccard_coef_int


class NeuralNetwork(object):

    def __init__(self, model):
        self.model = model
        self.x = None
        self.y = None
        self.x_val = None
        self.y_val = None
        self.x_test = None


    def init_data(self):
        self.x = self.create_image_dataset(TRAIN_INPUT_DATA_PATH)
        self.y = self.create_image_dataset(TRAIN_OUTPUT_DATA_PATH)
        self.y = self.modify_y_set(self.y)

        self.x_val = self.create_image_dataset(VALIDATION_INPUT_DATA_PATH)
        self.y_val = self.create_image_dataset(VALIDATION_OUTPUT_DATA_PATH)
        self.y_val = self.modify_y_set(self.y_val)
        #
        # self.x_test = self.create_image_dataset(TEST_INPUT_DATA_PATH)

    def create_image_dataset(self, directory):
        dataset = []
        filenames = os.listdir(directory)
        for f in filenames:
            if f.endswith(IMAGE_FORMAT):
                image = tiff.imread(directory + f)
                if len(image) == IMAGE_SIZE:
                    dataset.append(image)
        dataset = np.array(dataset, np.float16) #/ 1024.
        return dataset


    # updates y set that each low level element is included in array
    def modify_y_set(self, y):
        new_y = y.reshape(len(y), len(y[0]), len(y[0][0]), 1)
        print(len(new_y), len(new_y[0]), len(new_y[0][0]), len(new_y[0][0][0]))
        return new_y


    def train_network_generator(self):
        save_weights = callbacks.ModelCheckpoint("../data/weights.{epoch:02d}-{val_loss:.3f}.hdf5")
        change_learning_rate = LearningRateScheduler(self.__lr_scheduler)
        self.model.fit_generator(
            self.batch_generator(batch_size=16),
            epochs=20,
            verbose=1,
            validation_data=(self.x_val, self.y_val),
            callbacks=[save_weights, change_learning_rate],
            steps_per_epoch=1778 # it is N train images divided by batch size
            )
        self.model.save_weights("../data/weights.h5")

    def train_network(self):
        save_wights = callbacks.ModelCheckpoint("../data/weights.{epoch:02d}-{val_loss:.3f}.hdf5")
        # change_learning_rate = LearningRateScheduler(self.__lr_scheduler)

        history = self.model.fit(self.x, self.y, batch_size=12, epochs=40, validation_data=(self.x_val, self.y_val),
                  callbacks=[save_wights])
        self.__plot_history(history)
        self.model.save_weights("../data/weights.h5")

    def batch_generator(self, batch_size):
        train = sorted(os.listdir(TRAIN_INPUT_DATA_PATH))
        train_mask = sorted(os.listdir(TRAIN_OUTPUT_DATA_PATH))

        print(train)
        print(train_mask)

        for file in train:
            x = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
            y = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE))
            for i in range(batch_size):
              #  random_image = random.randint(0, self.x.shape[0] - 1)
                if file.endswith(IMAGE_FORMAT):
                    x[i] = tiff.imread(TRAIN_INPUT_DATA_PATH + file)
                    y[i] = tiff.imread(TRAIN_OUTPUT_DATA_PATH + file)

            new_y = y.reshape(len(y), len(y[0]), len(y[0][0]), 1)
            yield x, new_y


    def train_network_kfold_validation(self, folds_num=5):
        skf = KFold(n_splits=folds_num, shuffle=True)
        i = 0
        for train, test in skf.split(self.x):
            callback = callbacks.ModelCheckpoint("../data/" + str(i) + "-weights.{epoch:02d}-{val_loss:.3f}.hdf5")
            change_learning_rate = LearningRateScheduler(self.__lr_scheduler)
            print("%s %s" % (train.shape, test.shape))

            self.model.fit(self.x[train], self.y[train], batch_size=28, epochs=40, callbacks=[callback, change_learning_rate],
                      validation_data=(self.x[test], self.y[test]))
            self.model.save_weights("../data/weights.h5")
            i+=1

    def __lr_scheduler(self, epoch):
        if epoch == 9:
            curr_lr = K.get_value(self.model.optimizer.lr)
            curr_lr /= 10
            K.set_value(self.model.optimizer.lr, curr_lr)
        print(K.get_value(self.model.optimizer.lr))
        return K.get_value(self.model.optimizer.lr)

    def __plot_history(self, history):
        # history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        plt.savefig('../data/accuracy.png')
        plt.clf()
        # history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        plt.savefig('../data/loss.png')
        plt.clf()
        # history for jaccard coef
        plt.plot(history.history['jaccard_coef'])
        plt.plot(history.history['val_jaccard_coef'])
        plt.title('Jaccard coef')
        plt.ylabel('jaccard_coef')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        plt.savefig('../data/jaccard_coef.png')
        plt.clf()


    def test_network(self, weights):
        self.model.load_weights(weights)
        predictions = self.model.predict(self.x_test, batch_size=10)
        self.convert_predictions_by_threshold(predictions, 0.6)
        return predictions

    def convert_predictions_by_threshold(self, img, threshold):
        for x in np.nditer(img, op_flags=['readwrite']):
            if x > threshold:
                x[...] = 1.0
            else:
                x[...] = 0.0