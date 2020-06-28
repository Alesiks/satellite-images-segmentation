import segmentation_models as sm

from app.net.neuralnetwork import NeuralNetwork

SM_FRAMEWORK='tf.keras'
BACKBONE = 'resnet34'

def func():
    print(sm.__version__)
    sm.set_framework(SM_FRAMEWORK)
    preprocess_input = sm.get_preprocessing(BACKBONE)


    nn = NeuralNetwork("stub model")
    # nn.copyAndUnzip("/home/ales/programming/satellite-images-segmentation/data/train.zip", "/home/ales/")
    # nn.copyAndUnzip("/home/ales/programming/satellite-images-segmentation/data/validation.zip", "/home/ales/")


    nn.init_data()
    # load your data
    x_train, y_train, x_val, y_val = nn.x, nn.y, nn.x_val, nn.y_val
    print("sizes: " + str(len(x_train)) + ", " + str(len(y_train)) + " " + str(len(x_val)) + " " + str(len(y_val)))


    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    # fit model
    # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=16,
       epochs=1,
       validation_data=(x_val, y_val),
    )

func()