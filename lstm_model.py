
import keras.layers as layers
from keras.models import Sequential
from keras.layers import Dense,Embedding,TimeDistributed,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization


def build_lstm_with_layer(width=33, height=250, nb_classes=5, number_hidden_layers=1,
                          activation= None, batch_norm=False):
    print('Build LSTM RNN model ...')
    input_shape = (height, width)
    print('Input Shape ' + str(input_shape))
    model = Sequential()
    if activation is None:
        model.add(LSTM(units=128, return_sequences=True, input_shape=(250,33)))
    elif number_hidden_layers > 1:
        model.add(LSTM(128, activation=activation, return_sequences=True, input_shape=(250, 33)))
    else:
        model.add(LSTM(128, activation=activation, return_sequences=False, input_shape=(250, 33)))
    if batch_norm:
        model.add(BatchNormalization())
    for x in range(number_hidden_layers - 1):
        model.add(LSTM(units=128, return_sequences=True, input_shape=(250, 33)))
        if batch_norm:
            model.add(BatchNormalization())
    if number_hidden_layers > 1:
        model.add(LSTM(units=128, return_sequences=False))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dense(units=nb_classes, activation='softmax'))
    model.summary()
    return model


if __name__ == '__main__':
    build_lstm_with_layer(number_hidden_layers=2, batch_norm=True, activation='sigmoid')