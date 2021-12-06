import unicodedata
import re
import torch
from torchmetrics import Metric
from nlp_metrics.cocval_evalution import eval_nlp_scores
import statistics
from typing import Any, Callable, Optional
from transformers import BertTokenizer

MAX_LENGTH = 20

SOS_token = 2
EOS_token = 1
PAD_token = 0


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'PAD': 0, 'EOS': 1, 'SOS': 2}
        self.word2count = {}
        self.index2word = {0: 'PAD', 1: 'EOS', 2: 'SOS'}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)


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


# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"(['])", r" \1", s)
    s = re.sub(r"[^' 'a-z']+", r'', s)

    return s


class nlp_metric_bert(Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.add_state("preds", default=torch.empty((0, MAX_LENGTH), dtype=torch.int16), dist_reduce_fx="cat")
        self.add_state("target", default=torch.empty((0), dtype=torch.int16), dist_reduce_fx="cat")

        self.bleu1 = torch.tensor(0, dtype=torch.float32)
        self.bleu2 = torch.tensor(0, dtype=torch.float32)
        self.bleu3 = torch.tensor(0, dtype=torch.float32)
        self.bleu4 = torch.tensor(0, dtype=torch.float32)
        self.cider = torch.tensor(0, dtype=torch.float32)
        self.meteor = torch.tensor(0, dtype=torch.float32)
        self.rougel = torch.tensor(0, dtype=torch.float32)
        self.harmonice = torch.tensor(0, dtype=torch.float32)

    def update(self, new_preds, new_target):
        self.preds = torch.cat((self.preds, new_preds), dim=0)
        self.target = torch.cat((self.target, new_target), dim=0)

    def compute(self, val):
        total_text_generated = []
        total_text_reference = []
        # TODO: Try if this works without cpu/numpy/tolist
        generated_tokens = torch.reshape(self.preds, [self.preds.shape[0] * self.preds.shape[1],
                                                      self.preds.shape[2]]).tolist()
        target = torch.reshape(self.target,
                               [self.target.shape[0] * self.target.shape[1]]).tolist()
        print('length')
        print(len(target))
        print(len(generated_tokens))
        generate_converted = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for di in range(0, len(generated_tokens)):
            total_text_generated.append([generate_converted[di]])
            total_text_reference.append(val[target[di]])

        metrics_dict = eval_nlp_scores(total_text_generated, total_text_reference)
        self.bleu1 = torch.tensor(metrics_dict['Bleu_1'][0])
        self.bleu2 = torch.tensor(metrics_dict['Bleu_2'][0])
        self.bleu3 = torch.tensor(metrics_dict['Bleu_3'][0])
        self.bleu4 = torch.tensor(metrics_dict['Bleu_4'][0])

        self.cider = torch.tensor(metrics_dict['CIDEr'][0])
        self.meteor = torch.tensor(metrics_dict['METEOR'][0])
        self.rougel = torch.tensor(metrics_dict['ROUGE_L'][0])
        del total_text_generated, total_text_reference
        self.harmonice = torch.tensor(statistics.harmonic_mean([metrics_dict['Bleu_4'][0], metrics_dict['METEOR'][0]]))

        return self.harmonice
