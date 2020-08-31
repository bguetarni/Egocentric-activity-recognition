import shutil
import os
import numpy as np
import cv2
import pandas as pd
import tarfile
from epic_downloader import EpicDownloader
from tensorflow.keras import layers

HOME = os.path.expanduser('~')
DATA_PATH = '/data/EPIC-KITCHENS/'
LABELS_URL = 'https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-55-annotations/master/EPIC_train_action_labels.csv'
LABELS_PATH = HOME + '/labels.csv'
k = 4
batch_size = 32


def get_frames(id_min, id_max, data, path, resize):
    # computes step to apply
    if k <= (id_max - id_min) + 1:
        step = (id_max - id_min) // (k - 1)
    else:
        step = 1

    # extract frames uniformly in [id_min, id_max] and process it
    x = None
    for idx in range(id_min, id_max + 1, step):
        img = np.array(cv2.imread(path + data[idx]), dtype='float32')
        if resize:
            img = cv2.resize(img, (250, 250))
        if x is None:
            x = np.expand_dims(process_image(img), axis=0)
        else:
            x = np.append(x, np.expand_dims(process_image(img), axis=0), axis=0)
    return x


def process_image(x):
    # normalize the image with (mean, std) = (0, 1)
    rescale = layers.experimental.preprocessing.Rescaling(scale=1./255)
    normalize = layers.experimental.preprocessing.Normalization()
    x = rescale(x)
    normalize.adapt(x)
    return normalize(x)


def create_training_set(video_id, resize=False):
    labels = pd.read_csv(LABELS_PATH)
    frames_path = DATA_PATH + video_id + '/'
    frames_list = os.listdir(frames_path)
    frames_list.sort()
    line = 0

    # go next line until don't find the video ID
    while labels.at[line, 'video_id'] != video_id and line < labels.shape[0]:
        line += 1

    # loop over all the action in the video
    X, Y_verb, Y_noun = None, [], []
    while labels.at[line, 'video_id'] == video_id and line < labels.shape[0]:
        # retrieve data
        id_min, id_max = labels.at[line, 'start_frame'], labels.at[line, 'stop_frame']
        x = get_frames(id_min, id_max, frames_list, frames_path, resize)
        verb = np.zeros(125)
        verb[labels.at[line, 'verb_class']] = 1
        noun = np.zeros(352)
        noun[labels.at[line, 'noun_class']] = 1

        # store the data
        if X is None:
            X = np.expand_dims(x, axis=0)
        else:
            X = np.append(X, np.expand_dims(x, axis=0), axis=0)
        Y_verb.append(verb)
        Y_noun.append(noun)

        # go next line
        line += 1
    return X, [np.array(Y_verb, dtype='float32'), np.array(Y_noun, dtype='float32')]


def download_video_set(participants_id):
    # download TAR files from URL
    print('Downloading images')
    downloader = EpicDownloader(base_output='/data')
    downloader.download(what=['rgb_frames'], participants=participants_id, splits=['train'], challenges=['ar'],
                        epic55_only=True)

    # extract images
    print('Extracting images')
    for folder in os.listdir(DATA_PATH):
        for file in os.listdir(DATA_PATH + folder + '/rgb_frames/'):
            tar_file = tarfile.open(DATA_PATH + folder + '/rgb_frames/' + file)
            print('	Extracting {} to {}'.format(file, DATA_PATH + file[:6]))
            tar_file.extractall(DATA_PATH + file[:6])
        shutil.rmtree(DATA_PATH + folder, ignore_errors=True)
