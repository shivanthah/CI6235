from __future__ import print_function

''' 
Preprocess audio
'''
import numpy as np
import librosa
import librosa.display
import os
import re
from sklearn.model_selection import train_test_split

type_list = ['eairh', 'eh', 'heh', 'neh', 'owh']


def get_list_of_files(dir_name):
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        if entry.startswith('.'):
            continue

        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            if full_path.endswith(".wav"):
                all_files.append(full_path)

    return all_files


def save(features, labels, name, outpath):
    features_train, features_test, label_train, label_test = train_test_split(features, labels, test_size=0.2,
                                                                              random_state=1)
    features_train, features_val, label_train, label_val = train_test_split(features_train, label_train,
                                                                            test_size=0.2, random_state=1)
    with open(outpath + name + '_features_train.npy', 'wb') as f:
        np.save(f, features_train)
    with open(outpath + name + '_features_val.npy', 'wb') as f:
        np.save(f, features_val)
    with open(outpath + name + '_features_test.npy', 'wb') as f:
        np.save(f, features_test)
    with open(outpath + name + '_label_train.npy', 'wb') as f:
        np.save(f, label_train)
    with open(outpath + name + '_label_test.npy', 'wb') as f:
        np.save(f, label_test)
    with open(outpath + name + '_label_val.npy', 'wb') as f:
        np.save(f, label_val)


def compute_melgram(file_path):
    """ Compute a mel-spectrogram and returns it in a shape of (96,100, 1), where
       96 == #mel-bins and 300 == #time frame
    """
    print('computing mel-spetrogram for audio: ', file_path)

    # mel-spectrogram parameters
    sampling_rate = 12000
    n_fft = 512
    n_mels = 96
    hop_length = 256
    duration_in_seconds = 6.4 # to make it 1366 frame (300 = 12000 * 6 / 256)

    src, sr = librosa.load(file_path, sr=sampling_rate)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(duration_in_seconds * sampling_rate)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(duration_in_seconds * sampling_rate) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=sampling_rate, hop_length=hop_length,
                        n_fft=n_fft, n_mels=n_mels) ** 2,
                ref=1.0)
    ret = np.expand_dims(ret, axis=2)

    return ret


def preprocess_dataset_melgram(inpath="Dataset/", outpath="Pre-process-dataset/",name="melgram_feature"):
    """ Compute a preprocess_dataset_melgram
    """
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    files = get_list_of_files(inpath)
    labels = np.zeros((len(files), len(type_list)), dtype=np.float64)
    log_specgrams = []
    for idx, in_file_name in enumerate(files):

        splits = re.split('[ /]', in_file_name)
        label = splits[1]
        file_name = splits[2]
        index = type_list.index(label)
        labels[idx, index] = 1

        audio_path = inpath + label + '/' + file_name
        log_specgrams.append(compute_melgram(audio_path))

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), 96, 301, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)

    save(features,labels, name, outpath)


def preprocess_dataset_mfcc(inpath="Dataset/", outpath="Pre-process-dataset/",name="mfcc_feature"):
    """ Compute a preprocess_dataset_mfcc
    """
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    files = get_list_of_files(inpath)

    print_every = 20
    time_series_length = 250

    features = np.zeros((len(files), time_series_length, 33), dtype=np.float64)
    labels = np.zeros((len(files), len(type_list)), dtype=np.float64)

    for idx, in_file_name in enumerate(files):
        splits = re.split('[ /]', in_file_name)
        label = splits[1]
        file_name = splits[2]
        index = type_list.index(label)
        labels[idx, index] = 1

        audio_path = inpath + label + '/' + file_name
        y, sr = librosa.load(audio_path)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features[idx, :, 0:13] = mfcc.T[0:time_series_length, :]
        features[idx, :, 13:14] = spectral_center.T[0:time_series_length, :]
        features[idx, :, 14:26] = chroma.T[0:time_series_length, :]
        features[idx, :, 26:33] = spectral_contrast.T[0:time_series_length, :]
        if 0 == idx % print_every:
            print("Extracted features audio track %i of %i." % (idx + 1, len(files)))

    save(features, labels, name, outpath)


if __name__ == '__main__':
    preprocess_dataset_melgram()
    preprocess_dataset_mfcc()

