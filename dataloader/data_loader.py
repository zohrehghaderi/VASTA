import numpy as np
from torch.utils.data import Dataset
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import mmcv
import cv2

PAD_token = 0


def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)


class Video_Caption_Loader(Dataset):

    def __init__(self, dataset, path, type, vocab, max_length, config, adaptive=True):

        self.dataset = dataset
        self.confige = config
        self.path = path
        if isinstance(config, str):
            self.cfg = mmcv.Config.fromfile(config)
        self.transformer_video = self.cfg.transformer_video
        self.transformer_video = Compose(self.transformer_video)
        self.adaptive = adaptive
        self.map_class = np.load(self.path + '/tag.npy').tolist()
        self.vocab = vocab
        self.max_length = max_length
        self.type = type

    def __getitem__(self, index):
        """Returns one data pair (images and caption)."""
        video_name = self.dataset['video_name'][index]
        caption = self.dataset['caption'][index]

        video_file = self.path + '/videos/' + video_name + self.type
        if self.adaptive:
            frame_index_path = self.path + '/index_32/' + video_name + '.npy'
            frame_index = np.load(frame_index_path)
        else:

            cap = cv2.VideoCapture(video_file)
            time_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = np.linspace(0, time_depth - 1, num=32, dtype=int)

        data = dict(
            filename=video_file,
            label=-1,
            start_index=0,
            modality='RGB',
            frame_inds=frame_index,
            clip_len=32,
            num_clips=1

        )
        data = self.transformer_video(data)
        images = collate([data], samples_per_gpu=1)['imgs']
        images = images.squeeze(0)

        tag_file = self.path + '/tag/' + video_name + '.npy'
        lbl = np.load(tag_file)
        list_label = lbl.tolist()
        label = np.zeros(768, dtype=np.float16)

        i = 0
        while i < len(list_label):
            if list_label[i] in self.map_class:
                label[self.map_class.index(list_label[i])] = 1
            i = i + 1
        vocab = self.vocab

        if type(caption) != int:
            s_caption = caption.split()
            split_caption = s_caption[:self.max_length - 1]
            split_caption.append('</s>')
            for i in range(0, self.max_length - split_caption.__len__()):
                split_caption.append('<pad>')

            caption_encode = convert_list_to_string(split_caption)
            target = vocab.encode(caption_encode)
        else:
            target = caption

        return images, target, label

    def __len__(self):
        return self.dataset['video_name'].__len__()
