import glob, os
import csv
import pickle
import torchtext
from torchnlp.encoders.text import WhitespaceEncoder
import unicodedata
import re

SOS_token = 2
EOS_token = 1
PAD_token = 0


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"(['])", r" \1", s)
    s = re.sub(r"[^' 'a-z']+", r'', s)

    return s


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


class Msvd_dataset():
    def __init__(self, path):
        self.Dataset_path = path
        self.type = '.avi'

    def read_vocab(self):
        with open(self.Dataset_path + '/MSVD_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
            f.close()
        return vocab

    def read_video_with_caption(self):
        with open(self.Dataset_path + '/train_data.pickle', 'rb') as f:
            train = pickle.load(f)
        for i in range(train['caption'].__len__()):
            train['caption'][i] = normalizeString(train['caption'][i])

        with open(self.Dataset_path + '/val_data.pickle', 'rb') as f1:
            val = pickle.load(f1)

        with open(self.Dataset_path + '/test_data.pickle', 'rb') as f2:
            test = pickle.load(f2)
        return train, val, test



    def read_test(self):
        with open(self.Dataset_path + '/full_test.pickle', 'rb') as f1:
            test_data = pickle.load(f1)
        return test_data

    def read_val(self):

        with open(self.Dataset_path + '/full_val.pickle', 'rb') as f1:
            val_data = pickle.load(f1)
        return val_data

