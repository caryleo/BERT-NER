# BERT-NER

**This project developed on the work of [weizhepei/BERT-NER](https://github.com/weizhepei/BERT-NER). A implement of BERT+CRF and a new dataset CNER is added.**

Here is the original project introduction:
This project implements a solution to the "X" label issue (e.g., [#148](https://github.com/huggingface/pytorch-transformers/issues/148), [#422](https://github.com/huggingface/pytorch-transformers/issues/422)) of NER task in Google's BERT [paper](https://arxiv.org/pdf/1810.04805.pdf), and is developed mostly based on lemonhu's [work](https://github.com/lemonhu/NER-BERT-pytorch) and bheinzerling's [suggestion](https://github.com/huggingface/pytorch-transformers/issues/64#issuecomment-443703063).


## Dataset

- Chinese: [MSRA](http://sighan.cs.uchicago.edu/bakeoff2006/), which is [reported](https://github.com/lemonhu/NER-BERT-pytorch/issues/9) to be incomplete. A complete version can be found [here](https://github.com/buppt/ChineseNER/tree/master/data/MSRA).
- English: [CONLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)、NCER（**加一下链接**）

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.0. The main requirements are:

- nltk
- tqdm
- pytorch >= 1.3.1
- 🤗transformers == 2.2.2

## Basic Model

We provide three datasets and three models in this project. 

**Dataset**

CONLL-2013: A common dataset for English NER.

MSRA: A common dataset for Chinese NER.

NCER: Another dataset for English NER with more categories than CONLL.

Specify the parameters on the command line use --dataset conll\msra\ncer. Take msra for example 

```shell
  python xxx --dataset=msra
```

**Model**

Linear: The original BERT implement which add a Linear layer after a BERT. 

CRF1: The first version of our BERT+CRF, which only support batch_size=1.

CRF2: The second version of our BERT+CRF, which can work on random size of batch_size. CRF2 is much more faster than CRF1.

Specify the parameters on the command line use --model linear\crf1\crf2. Take crf2 for example 

```shell
  python xxx --dataset=msra --model crf2
```

## Quick Start

- **Download and unzip the [Chinese](https://drive.google.com/open?id=1eo4HwLrix-Zbiu5qSl29cc_3uUGKF9yZ) ([English](https://drive.google.com/open?id=1iS2Zu93ecmvxYlIrAxv0A5qG_r0o8Exy)) NER model weights under `experiments/msra(conll)/`, then  run:**

  ```shell
  python build_dataset_tags.py --dataset=msra --model crf2
  python interactive.py --dataset=msra --model crf2
  ```

  **to try it out and interact with the pretrained NER model**.

## Usage

1. **Get BERT model for PyTorch**

   There are two ways to get the pretrained BERT model in a PyTorch dump for your experiments :

   - **[Automatically] Download the specified pretrained BERT model provided by *huggingface🤗***

   - **[Manually] Convert the TensorFlow checkpoint to a PyTorch dump**

     - Download the Google's BERT pretrained models for Chinese  **[(`BERT-Base, Chinese`)](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)** and English **[(`BERT-Base, Cased`)](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**. Then decompress them under `pretrained_bert_models/bert-chinese-cased/` and `pretrained_bert_models/bert-base-cased/` respectively. More pre-trained models are available [here](https://github.com/google-research/bert#pre-trained-models).

     - Execute the following command,  convert the TensorFlow checkpoint to a PyTorch dump as huggingface [suggests](https://huggingface.co/pytorch-transformers/converting_tensorflow_models.html). Here is an example of the conversion process for a pretrained `BERT-Base Cased` model.

       ```shell
       export TF_BERT_MODEL_DIR=/full/path/to/cased_L-12_H-768_A-12
       export PT_BERT_MODEL_DIR=/full/path/to/pretrained_bert_models/bert-base-cased
        
       transformers bert \
         $TF_BERT_MODEL_DIR/bert_model.ckpt \
         $TF_BERT_MODEL_DIR/bert_config.json \
         $PT_BERT_MODEL_DIR/pytorch_model.bin
       ```

     - Copy the BERT parameters file `bert_config.json` and dictionary file `vocab.txt` to the directory `$PT_BERT_MODEL_DIR`.

       ```
       cp $TF_BERT_MODEL_DIR/bert_config.json $PT_BERT_MODEL_DIR/config.json
       cp $TF_BERT_MODEL_DIR/vocab.txt $PT_BERT_MODEL_DIR/vocab.txt
       ```

2. **Build dataset and tags**

   if you use CONLL-2003 dataset and model CRF for example, just specify the parameters on the command line. The dataset  can be conll, msra or cner and  model can be linear, crf,crf2. 

   ```shell
   python build_dataset_tags.py --dataset conll --model crf2
   ```

   It will extract the sentences and tags from `train_bio`, `test_bio` and `val_bio`(if not provided, it will randomly sample 5% data from the `train_bio` to create `val_bio`). Then split them into train/val/test and save them in a convenient format for our model, and create a file `tags.txt` containing a collection of tags.

3. **Set experimental hyperparameters**

   We created directories with the same name as datasets under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like

   ```json
   {
       "full_finetuning": true,
       "max_len": 180,
       "learning_rate": 5e-5,
       "weight_decay": 0.01,
       "clip_grad": 5,
   }
   ```

   For different datasets, you will need to create a new directory under `experiments` with  `params.json`.

4. **Train and evaluate the model**

   If you use CONLL dataset and run on crf2 model, just run 

   ```shell
   python train.py --dataset=conll --model=crf2
   ```

   You can alse use other parameters discribed in the **Basic Model** part.

   A proper pretrained BERT model will be automatically chosen according to the language of the specified dataset. It will instantiate a model and train it on the training set following the hyper-parameters specified in `params.json`. It will also evaluate some metrics on the development set.
   
   **NOTE:** You should build the dataset with the same parameters before train. 

5. **Evaluation on the test set**

   Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set.

   If you use CONLL dataset and run on crf2 model, just run 

   ```shell
   python evaluate.py --dataset=conll --model=crf2
   ```


​       You can alse use other parameters discribed in the **Basic Model** part.