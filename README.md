# VASTA

This repository contains the official implementation to reproduce the results for our VASTA: Diverse Video Captioning by Adaptive Spatio-temporal Attention Paper accpted GCPR 2022 [link](https://link.springer.com/chapter/10.1007/978-3-031-16788-1_25). 

## Abstract
To generate proper captions for videos, the inference needs to identify relevant concepts and pay attention to the spatial relationships between them as well as to the temporal development in the clip. Our end-to-end encoder-decoder video captioning framework incorporates two transformer-based architectures, an adapted transformer for a single joint spatio-temporal video analysis as well as a self-attention-based decoder for advanced text generation. Furthermore, we introduce an adaptive frame selection scheme to reduce the number of required incoming frames while maintaining the relevant content when training both transformers. Additionally, we estimate semantic concepts relevant for video captioning by aggregating all ground truth captions of each sample. Our approach achieves state-of-the-art results on the MSVD, as well as on the large-scale MSR-VTT and the VATEX benchmark datasets considering multiple Natural Language Generation (NLG) metrics. Additional evaluations on diversity scores highlight the expressiveness and diversity in the structure of our generated captions.

![Alt Text](https://github.com/GCPR36/GCPR2022_submission_36/blob/master/VASTA-Model.gif)


## Table of Contents

It contains the following sections:
1. Data download
2. Requirements and Setup
3. Training the models
4. Pre-trained checkpoints
5. Evaluating the trained models.

To start you need to clone this repository and `cd` into the root directory.

```bash
git clone https://github.com/zohrehghaderi/VASTA.git
cd VASTA
```

## Data Download
We show results on two datasets MSVD and MSR-VTT. We provide output of our adaptive frame selection method in `data\dataset_name\index_32` and ralated lables in sematics network are in `data\dataset_name\tag`. As well, normalized captions are `data\dataset_name\file.pickle`. For using this code, it is important to download videos of both dataset and put in `data\dataset_name\videos`. For example, MSVD dataset is following this tree:
```bash

data
  |--MSVD
       |--index_32    \\ output adaptive frame selection 
       |--tag          \\ extracted tag for semantics network
       |--videos        \\ video
       |-MSVD_vocab.pkl  \\ word dictionary 
       |-full_test.pickle \\ to evalute NLP Metrics on test data
       |-full_val.pickle   \\ to evalute NLP Metrics on validation data
       |-tag.npy            \\ tag dictionary
       |-test_data.pickle    \\ test video name and related caption 
       |-train_data.pickle    \\ train video name and related caption
       |-val_data.pickle       \\ val video name and related caption
```
### MSVR
To download MSVD, follow this [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)

### MSR-VTT
To download MSR-VTT, follow this [link](https://www.mediafire.com/folder/h14iarbs62e7p/shared)


## Requirements and Setup
To run our coda, create a conda environment with this command.

```bash
conda env update -f environment.yml -n TED
conda activate TED
```
This will install all dependencies described in our `environment.yml` file.

### Swin-B
To download the weights of the Swin-B network, refer to [Link](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth)  and then put in `checkpoint/swin`


### NLP Metrics
In this repository, [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) is emplyed into `nlp_metrics` folder to evaluate validition and test data.

## Training the models
We show several models in our paper (AFS-Swin-Bert-semantics, UFS-Swin-Bert-semantics, AFS-Swin-Bert, UFS-Swin-Bert) with --afs and --semantics being True or False. The latter ones are ablations.
 `DATASET_NAME` is msvd or msrvtt.


### <best-model-replace-me>
To train our best <AFS-Swin-Bert-semantics> TED-VC model use this command:
```bash
 python main.py --afs=True  --dataset=DATASET_NAME --semantics=True --ckp_semantics=checkpoint/semantics_net/DATASET_NAME/semantics.ckpt 
```
### <ufs-model-replace-me>
To train our best <UFS-Swin-Bert-semantics> TED-VC model which does not use Adaptive Frame Selection (AFS) use this command:
```bash
 python main.py --afs=False  --dataset=DATASET_NAME --semantics=True --ckp_semantics=checkpoint/semantics_net/DATASET_NAME/semantics.ckpt 

```
  
<Continue with all other models>


## Pre-trained checkpoints
Additionally, you can find pre-trained checkpoints of our model here

| Model Name              | Dataset   | Link       |
|-------------------------|-----------|------------|
| AFS-Swin-Bert-semantics | MSVD      | [link](https://drive.google.com/file/d/11Qr7Ivi4H90HgsBqo1nf8JXRE1xXea5d/view?usp=sharing)|
| UFS-Swin-Bert-semantics | MSVD      | [link](https://drive.google.com/file/d/1MynAFcjFqPWhI3y0pejK0aRFWSQvaEn6/view?usp=sharing)|
| AFS-Swin-Bert-semantics | MSRVTT    | [link](https://drive.google.com/file/d/12KIGYZ3orEeErDHzqTXLjIy9IW0A1Al_/view?usp=sharing)| 
| UFS-Swin-Bert-semantics | MSRVTT    | [link](https://drive.google.com/file/d/1lbriyNuIhWnMpi9cLDWJOfxHgOKmJ672/view?usp=sharing)|



## Evaluating the trained models.
### <best-model-replace-me>
To train our best TED-VC model use this command:
```bash
 python test.py --afs=True  --dataset=DATASET_NAME --semantics=True --bestmodel=LINK_BESTMODEL
```
for example for MSVD dataset:
  ```bash
 python test.py --afs=True  --dataset=msvd --semantics=True --bestmodel=bestmodel/msvd/AFSSemantics.ckpt
```



### <ufs-model-replace-me>
To train our best  TED-VC model which does not use Adaptive Frame Selection (AFS) use this command:
```bash
 python test.py --afs=False  --dataset=DATASET_NAME --semantics=True --bestmodel=LINK_BESTMODEL
```

 ### Result

This should produce the following results <copy from paper>:
 
|        Model Name       | Dataset | Bleu-4 | METEOR | CIDER | ROUGE-L |
|-------------------------|---------|--------|--------|-------|---------|
| AFS-Swin-Bert-semantics | MSVD    |  56.14 |  39.09 | 106.3 |  74.47  |
| UFS-Swin-Bert-semantics | MSVD    |  54.30 |  38.18 | 102.7 |  74.28  |
| AFS-Swin-Bert-semantics | MSRVTT  |  43.43 |  30.24 | 55.00 |  62.54  |
| UFS-Swin-Bert-semantics | MSRVTT  |  43.51 |  29.75 | 53.59 |  62.27  |

To train our best TED-VC model which does not use Adaptive Frame Selection (AFS) and semantics network use this command:
```bash
 python test.py --afs=False  --dataset=DATASET_NAME --semantics=False --bestmodel=LINK_BESTMODEL
```
## License
Note that this is a confidential code release only meant for the purpose of reviewing our submission.

## Citing VASTA
```
@inproceedings{ghaderi2022diverse,
  title={Diverse Video Captioning by Adaptive Spatio-temporal Attention},
  author={Ghaderi, Zohreh and Salewski, Leonard and Lensch, Hendrik PA},
  booktitle={DAGM German Conference on Pattern Recognition},
  pages={409--425},
  year={2022},
  organization={Springer}
}
```
## Acknowledgements
This readme is inspired by https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md.
