from __future__ import print_function
import os
import torch
import random
import argparse
import json
import spacy
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from datetime import datetime
# import model
from torchtext.data import TabularDataset, Field

spacy_en = spacy.load('en')
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def main(params):
    try:
        output_dir = os.path.join(params['outf'], datetime.strftime(datetime.now(), "%Y%m%d_%H%M"))
        os.makedirs(output_dir)
    except OSError:
	    pass

    if torch.cuda.is_available() and not params['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)
    IMG_PATH =  Field(sequential=False, use_vocab=False)

    fields = {'ans':('ans', LABEL), 'img_path':('img_path', IMG_PATH), 'question':('question', TEXT)}

    train, val = TabularDataset.splits(
                path=params['dataroot'], 
                train=params['input_train'], 
                validation=params['input_test'], 
                format='csv',
                skip_header=False, 
                fields=fields
                )

    print("Train data")
    print(train[0].__dict__.keys())
    print(train[0].ans, train[0].img_path, train[0].question)

    print("Validation data")
    print(val[0].__dict__.keys())
    print(val[0].ans, val[0].img_path, val[0].question)

    print("Building Vocabulary ..")
    TEXT.build_vocab(train, vectors='glove.6B.100d')
    vocab = TEXT.vocab

    print("Creating Embedding from vocab vectors ..")    
    # embed = nn.Embedding(len(vocab), params['nte'])
    # embed.weight.data.copy_(vocab.vectors)
    embed = nn.Embedding.from_pretrained(vocab.vectors)

    print("Text Embeddings are generated of size ", embed.weight.size())


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train', default='vqa_train.csv', help='input json file')
    parser.add_argument('--input_test', default='vqa_test.csv', help='input json file')
    parser.add_argument('--mapping_file', default='image_index.pkl', help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument('--embedding_output_path', default='./data/img_embedding.pkl', help='output pkl file with img features')

    parser.add_argument(
        '--dataroot', default='./data/', help='path to dataset')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument(
        '--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument(
        '--imageSize',
        type=int,
        default=224,
        help='the height / width of the input image to network')
    parser.add_argument(
        '--nte',
        type=int,
        default=100,
        help='the size of the text embedding vector')
    parser.add_argument(
        '--nif',
        type=int,
        default=2048,
        help='the size of the image feature vector')
    parser.add_argument(
        '--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument(
        '--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument(
        '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument(
        '--outf',
        default='./output/',
        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument(
        '--eval', 
        action='store_true', 
        help="choose whether to train the model or show demo")
    
    args = parser.parse_args()
    params = vars(args)

    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))

    main(params)
