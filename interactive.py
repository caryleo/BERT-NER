"Evaluate the model"""
import os
import nltk
import torch
import random
import logging
import argparse
import numpy as np
import utils as utils
from metrics import get_entities
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging
from CRFTagger import BertForCRFTagging
from CRFTagger2 import BertForCRFTagging2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='msra', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--model', default='linear', choices=['linear', 'crf', 'crf2'], help="The Model we want to use")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def interAct(model, data_iterator, params,seq_len,mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    idx2tag = params.idx2tag

    batch_data, batch_token_starts = next(data_iterator)
    batch_masks = batch_data.gt(0)

    if params.model == 'linear':
        batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[
            0]  # shape: (batch_size, max_len, num_labels)
        batch_output = batch_output.detach().cpu().numpy()  # must sent to cpu to use numpy
        batch_output_argmax = np.argmax(batch_output, axis=2)
    elif params.model == 'crf':
        batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[
            1]  # list of the tag index of each word e./[0,0,1,2,1],denoting the index of a tag.
        batch_output_argmax = [batch_output]  # only for batch_size=1,to be modified later.

    elif params.model == 'crf2':
        outputs = \
            model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)
        logits = outputs[0]
        masks=torch.full((1,seq_len),True).to(params.device)
        # masks = batch_tags.gt(-1)
        batch_output = model.crf.decode(logits, masks, pad_tag=-1)[0]
        batch_output = batch_output.detach().cpu().numpy()
        batch_output_argmax = batch_output
    
    pred_tags = []
    pred_tags.extend([[idx2tag.get(idx) if idx != -1 else 'O' for idx in indices] for indices in batch_output_argmax])
    return(get_entities(pred_tags))


def bert_ner_init():
    args = parser.parse_args()
    tagger_model_dir = 'experiments/' + args.dataset

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    params.model = args.model

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset
    if args.dataset in ["conll", "wnr"]:
        bert_class = 'bert-base-cased'  # auto
        # bert_class = 'pretrained_bert_models/bert-base-cased/' # manual
    elif args.dataset in ["msra", "cner"]:
        bert_class = 'bert-base-chinese'  # auto
        # bert_class = 'pretrained_bert_models/bert-base-chinese/' # manual

    if params.model == 'crf':
        params.batch_size=1
    data_loader = DataLoader(data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1)

    # Load the model
    # Load the model
    if params.model == 'linear':
        model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    elif params.model == 'crf':
        model = BertForCRFTagging.from_pretrained(tagger_model_dir)
        model.load_option(params)
    elif params.model == 'crf2':
        model = BertForCRFTagging2.from_pretrained(tagger_model_dir)
    # model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    model.to(params.device)

    return model, data_loader, args.dataset, params

def BertNerResponse(model, queryString):    
    model, data_loader, dataset, params = model
    if dataset in ["msra", "cner"]:
        queryString = [i for i in queryString]
    elif dataset in ["conll", "wnr"]:
        queryString = nltk.word_tokenize(queryString)

    seq_len=len(queryString)

    with open('data/' + dataset + '/interactive/sentences.txt', 'w') as f:
        f.write(' '.join(queryString))

    inter_data = data_loader.load_data('interactive')
    inter_data_iterator = data_loader.data_iterator(inter_data, shuffle=False)
    result = interAct(model, inter_data_iterator, params,seq_len=seq_len)
    res = []
    for item in result:
        if dataset in ['msra']:
            res.append((''.join(queryString[item[1]:item[2]+1]), item[0]))
        elif dataset in ['conll']:
            res.append((' '.join(queryString[item[1]:item[2]+1]), item[0]))
    return res


def main():
    model = bert_ner_init()
    while True:
        query = input('Input:')
        if query == 'exit':
            break
        print(BertNerResponse(model, query))


if __name__ == '__main__':
    main()


    

