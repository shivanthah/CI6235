
from keras.optimizers import SGD
import numpy as np
from icc.cnn_model import build_cnn2
from keras.models import model_from_json

model_name = 'LSTM'


def eveluate(x_test,y_test):
    model = build_cnn2()
    model.load_weights("Model/CNN_2/cnn_weights-best.hdf5")

    # Keras optimizer defaults:
    # Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
    # RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
    # SGD    : lr=0.01, momentum=0., decay=0.
    opt = SGD()

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def evaluate_save_model(x_test,y_test):
    # load json and create model
    json_file = open('Model/CNN_2/CNN2_relu_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Model/CNN_2/CNN2_relu_model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    x = np.load('Pre-process-dataset/melgram_feature_features_test.npy')
    y = np.load('Pre-process-dataset/melgram_feature_label_test.npy')
    evaluate_save_model(x,y)