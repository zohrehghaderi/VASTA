# VASTA

This repository contains the official implementation to reproduce the results for our VASTA submission.


## Table of Contents

It contains the following sections:
1. Data download
2. Requirements and Setup
3. Training the models
4. Pre-trained checkpoints
5. Evaluating the trained models.

To start you need to clone this repository and `cd` into the root directory.

```bash
git clone https://github.com/ECCV7129/ECCV2022_submission_7129.git
cd ECCV2022_submission_7129
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

## Acknowledgements
This readme is inspired by https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md.
