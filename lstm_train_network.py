import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from itertools import product

from icc.lstm_model import build_lstm_with_layer


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def w_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
    return loss


# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
model_name = 'LSTM'
run = 'CW12'
log_dir = "logs/" + model_name + "/" + run + "/"
model_dir = "Model/" + model_name + "/" + run + "/"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(model_dir):
        os.mkdir(model_dir)

number_hidden_layers = 1
batch_norm = False
activation = 'tanh'
model_name = model_name + "_" + run + "_" + activation + "_" + str(number_hidden_layers)

x_train = np.load('Pre-process-dataset/mfcc_feature_features_train.npy')
y_train = np.load('Pre-process-dataset/mfcc_feature_label_train.npy')
x_val = np.load('Pre-process-dataset/mfcc_feature_features_val.npy')
y_val = np.load('Pre-process-dataset/mfcc_feature_label_val.npy')
batch_size = 20
nb_epochs = 100

print(x_train.shape)
print(y_train.shape)

tensorboard = TensorBoard(
  log_dir=log_dir,
  histogram_freq=0,  # How often to logs histogram visualizations
  embeddings_freq=0,  # How often to logs embedding visualizations
  update_freq='epoch')  # How often to write logs (default: once per epoch))

filepath= model_dir + "lstm_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]

model = build_lstm_with_layer(number_hidden_layers=number_hidden_layers,
                              batch_norm=batch_norm,
                              activation='tanh')

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam(lr=0.01)

#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

nb_classes = 5
wcc = w_categorical_crossentropy(np.ones((nb_classes, nb_classes)))

model.compile(loss=wcc, optimizer=opt, metrics=['accuracy'])


#type_list = ['eairh', 'eh', 'heh', 'neh', 'owh']
class_weight = {0: 2,
                1: 3,
                2: 1.5,
                3: 1,
                4: 4}
print("Training ...")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks_list,
                    validation_data=(x_val, y_val),class_weight=class_weight)

print(history.history.keys())
# list all data in history
print(history.history.keys())

# serialize weights to HDF5
model.save_weights(model_dir + model_name + "_model.h5")
print("Saved model to disk")

model_json = model.to_json()
with open(model_dir + model_name + "_model.json", "w") as json_file:
    json_file.write(model_json)
