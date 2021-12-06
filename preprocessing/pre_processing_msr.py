import pickle

MAX_LENGTH = 35
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



class Msr_dataset():
    def __init__(self, path):
        self.Dataset_path = path
        self.type = '.mp4'

    def read_vocab(self):
        with open(self.Dataset_path + '/MSR_VTT_vocab.pkl', 'rb') as f:
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


    def convert_list_to_string(org_list, seperator=' '):
        """ Convert list to string, by joining all item in list with given separator.
            Returns the concatenated string """
        return seperator.join(org_list)

    def read_test(self):
        with open(self.Dataset_path + '/full_test.pkl', 'rb') as f1:
            test_data = pickle.load(f1)
        return test_data['sentences']

    def read_val(self):

        with open(self.Dataset_path + '/full_val.pkl', 'rb') as f1:
            val_data = pickle.load(f1)
        return val_data['sentences']

