import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

from icc.cnn_model import build_cnn2

# Turn off TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
model_name = 'CNN_5'
activation = 'relu'
model_name = model_name + "_" + activation

x_train = np.load('Pre-process-dataset/melgram_feature_features_train.npy')
y_train = np.load('Pre-process-dataset/melgram_feature_label_train.npy')
x_val = np.load('Pre-process-dataset/melgram_feature_features_val.npy')
y_val = np.load('Pre-process-dataset/melgram_feature_label_val.npy')
batch_size = 5
nb_epochs = 50


tensorboard = TensorBoard(
  log_dir='logs/CNN_5',
  histogram_freq=0,  # How often to logs histogram visualizations
  embeddings_freq=0,  # How often to logs embedding visualizations
  update_freq='epoch')  # How often to write logs (default: once per epoch))

filepath="Model/CNN_5/cnn_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,tensorboard]

model = build_cnn2()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = SGD(lr=0.1)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
print("Training ...")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks_list,
                    validation_data=(x_val, y_val))


# serialize weights to HDF5
model.save_weights("Model/" + model_name + "_model.h5")
print("Saved model to disk")

model_json = model.to_json()
with open("Model/" + model_name + "_model.json", "w") as json_file:
    json_file.write(model_json)
