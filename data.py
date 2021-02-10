import argparse
import bisect
import os
import numpy as np
import pandas
import cv2


def get_images_indexes(id_min, id_max, k):
    """ Retieve the indexes of images sampled uniformly """

    idx_list = []
    if k > (id_max - id_min + 1):  # not enough images, use duplicates
        # construct a list of all the values
        to_sample = list(range(id_min, id_max + 1))
        
        # initialize the values list to all possible values and 0 for the rest
        idx_list = [0 for i in range(k)]
        idx_list[:len(to_sample)] = to_sample

        # fill the 0s of the list with duplicate values
        for i in range(k-len(to_sample)):

            # sample a random value in the list and store its index
            random_index = to_sample.pop(np.random.randint(0, len(to_sample)))
            j = idx_list.index(random_index)

            # save the values following
            tmp = idx_list[j+1:]

            # duplicate the value and re-write the saved values
            idx_list[j+1] = idx_list[j]
            idx_list[j+2:] = tmp[:(len(idx_list)-(j+2))]
    else:  # enough images, sample indexes uniformly
        step = (id_max - id_min)//k + 1
        idx_list = list(range(id_min, id_max + 1, step))
        if len(idx_list) < k:  # almost same as before, only no duplicate
            to_sample = list(range(id_min, id_max + 1))
            to_sample = list(set(to_sample) - set(idx_list))
            for i in range(k - len(idx_list)):
                random_index = to_sample.pop(np.random.randint(0, len(to_sample)))
                j = bisect.bisect_left(idx_list, random_index)
                tmp = idx_list[j:]
                idx_list += [0]
                idx_list[j] = random_index
                idx_list[j+1:] = tmp[:(len(idx_list)-(j + 1))]
    return idx_list


def get_frames(id_min, id_max, k, data, path):
    """ Retrieve the frames for the scpecified action """

    idx_list = get_images_indexes(id_min, id_max, k)
    x = []
    for i in idx_list:
        if i > len(data)-1:
            return np.NaN

        # read the image and convert it to RGB format
        buff = cv2.imread(path + data[i])
        img = np.array(cv2.cvtColor(buff, cv2.COLOR_BGR2RGB))

        # adapt the size
        img = cv2.resize(img, (256, 256))
        
        # add current image to the sequence
        x.append(img)
    
    return np.stack(x, axis=0)


def get_video_split(video_id):
    """ Look for split category of video """
    splits = pandas.read_csv('data/epic_100_splits.csv')
    
    # go next line until don't find the video ID
    line = 0
    while line < splits.shape[0] and splits.at[line, 'video_id'] != video_id:
        line += 1
    
    if line == splits.shape[0]:
        print('WARNING: video {} is not listed in epic_100_splits.csv'.format(video_id))
        return None
    else:
        if splits.at[line, 'ar_train']:
            return 'train'
        elif splits.at[line, 'ar_val']:
            return 'val'
        else:
            print('WARNING: video {} is for test'.format(video_id))
            return None


def create_video_data(path, video_id, k):
    """ Create the video numpy arrays (input,output) """
    
    # labels
    split = get_video_split(video_id)
    if split == 'train':
        labels = pandas.read_csv('data/EPIC_100_train.csv')
    elif split == 'val':
        labels = pandas.read_csv('data/EPIC_100_validation.csv')
    else:
        return None, None, None
    
    frames_path = path + video_id + '/'
    frames_list = os.listdir(frames_path)
    frames_list.sort()
    line = 0

    # go next line until it's video ID
    while line < labels.shape[0] and labels.at[line, 'video_id'] != video_id:
        line += 1

    # loop over all the actions in the video
    X, Y_verb, Y_noun = [], [], []
    while line < labels.shape[0] and labels.at[line, 'video_id'] == video_id:
        # retrieve the frames
        id_min, id_max = labels.at[line, 'start_frame'], labels.at[line, 'stop_frame']
        x = get_frames(id_min, id_max, k, frames_list, frames_path)
        if x is np.NaN:
            print('video {} not complete.'.format(video_id))
            return None, None, None
        if x is None:
            continue
        
        # EPICK-KITCHENS-100 has 97 verbs and 300 nouns
        verb, noun = np.zeros(97, dtype='uint8'), np.zeros(300, dtype='uint8')
        verb[labels.at[line, 'verb_class']] = 1
        noun[labels.at[line, 'noun_class']] = 1

        # add the data
        X.append(x)
        Y_verb.append(verb)
        Y_noun.append(noun)

        # go next line
        line += 1
    
    if len(X) < 1:
        return None, None, None
    else:
        return np.stack(X, axis=0), np.array(Y_verb), np.array(Y_noun)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, action='store', required=True, help='path to the folder containing the videos RGB images')
    parser.add_argument('--nb_frames', type=int, action='store', required=True, help='length of time-serie sequence')
    parser.add_argument('--out_path', type=str, action='store', required=True, help='path to save the created arrays')
    args = parser.parse_args()
    """
    data folder must be like:
        data\
            PXX_XX\
                    frame_0001.jpg
                    frame_0002.jpg
                    frame_0003.jpg
                    .
                    .
                    .
            PXX_XX\
                    frame_0001.jpg
                    frame_0002.jpg
                    frame_0003.jpg
                    .
                    .
                    .
            .
            .
            .

    out_path folder will be like:
        out_path\
                PXX_XX.npz
                PXX_XX.npz
                PXX_XX.npz
                .
                .
                .
    """

    videos_list = os.listdir(args.data)
    videos_list.sort()
    videos_already_done = os.listdir(args.out_path)
    dataset_size = 0
    for i, video in enumerate(videos_list):
        if video + '.npz' not in videos_already_done:
            print('\r{}/{} -> {} samples.'.format(i, len(videos_list), dataset_size), end='')
            X, Y_verb, Y_noun = create_video_data(args.data, video, args.nb_frames)
            if X is not None:
                dataset_size += X.shape[0]
                np.savez(args.out_path + '{}.npz'.format(video), x=X, y_verb=Y_verb, y_noun=Y_noun)
