from app.predition.img_dataset_prediction import DatasetPredictor

# preparer = ImagesPreparer()
# preparer.create_data_inria_aerial_images(300)

# net = NeuralNetwork()
# net.init_data()
# net.train_network()
# numpy.set_printoptions(threshold=numpy.nan)

# imgPrediction = ImagePrediction(["../data/weights.11-0.250.hdf5"])
# img = imgPrediction.predict_image("../../../machine learning data/NEW2-AerialImageDataset/train/images/austin1.tif")

tester = DatasetPredictor(["../data/weights.11-0.250.hdf5"])
tester.predictInriaAerialDataset()

