from __future__ import print_function
import os
import torch
import random
import argparse
import json
import spacy
import pickle as pkl
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
import model
from torchtext.data import TabularDataset, Field, Iterator

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

    SOS_token = '<sos>'
    EOS_token = '<eos>'

    TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, lower=True, batch_first=True, init_token=SOS_token, eos_token=EOS_token)
    # LABEL = Field(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True, batch_first=True, init_token='#', eos_token='$')
    IMG_IND =  Field(sequential=False, use_vocab=False, batch_first=True)

    fields = {'ans':('ans', TEXT), 'img_ind':('img_ind', IMG_IND), 'question':('question', TEXT)}

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
    print(train[0].ans, train[0].img_ind, train[0].question)

    print("Validation data")
    print(val[0].__dict__.keys())
    print(val[0].ans, val[0].img_ind, val[0].question)

    print("Building Vocabulary ..")
    TEXT.build_vocab(train, vectors='glove.6B.100d')
    vocab = TEXT.vocab

    print("Creating Embedding from vocab vectors ..")    
    # embed = nn.Embedding(len(vocab), params['nte'])
    # embed.weight.data.copy_(vocab.vectors)
    txt_embed = nn.Embedding.from_pretrained(vocab.vectors)
    print("Text Embeddings are generated of size ", txt_embed.weight.size())

    print("Loading Image embeddings ..")
    with open(params['image_embeddings'],'rb') as f:
            img_embs = pkl.load(f)['image_features']

    img_embed = nn.Embedding.from_pretrained(torch.FloatTensor(img_embs)) 

    print("Creating Encoder ..")
    encoder = model.Encoder(img_embed, txt_embed, params)
    print(encoder)
    
    print("Creating Decoder ..")
    decoder = model.Decoder(txt_embed, params)
    print(decoder)

    criterion = torch.nn.PairwiseDistance(keepdim=True)

    if params['cuda']:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])

    train_iter, val_iter = Iterator.splits((train, val), batch_sizes=(params['batch_size'], params['batch_size']))

    for epoch in range(params['niter']):
        encoder.train()
        decoder.train()
        for i, row in enumerate(train_iter):
            
            encoder.zero_grad()
            decoder.zero_grad()

            ans, img_ind, question = row.ans, row.img_ind, row.question
            target_length = ans.shape[1]
            # batch_size = ans.size(0)

            if params['cuda']:
                ans = ans.cuda()
                img_ind = img_ind.cuda()
                question = question.cuda()

            img_ind = Variable(img_ind)
            question = Variable(question)
            ans_embed = txt_embed(ans)

            context = encoder(img_ind, question)
            
            decoder_input = torch.tensor(ans[0][0])
            decoder_hidden = context

            if params['cuda']:
                decoder_input = decoder_input.cuda()

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, ans_embed[di])
                if decoder_input.item() == EOS_token:
                    break

            train_loss = criterion(pred_ans, ans_embed)
            train_loss.backward()
            encoder_optimizer.step()

            print('[%d/%d][%d/%d] train_loss: %.4f' %(epoch, params['niter'],i, len(train_iter), train_loss))

        # if epoch % 5 == 0:
        #     print('Calculating Validation loss')
        #     vqa_model.eval()
        #     avg_loss = 0
        #     for i, row in enumerate(val_iter):

        #         vqa_model.zero_grad()
        #         ans, img_ind, question = row.ans, row.img_ind, row.question
                
        #         batch_size = ans.size(0)

        #         if params['cuda']:
        #             ans = ans.cuda()
        #             img_ind = img_ind.cuda()
        #             question = question.cuda()

        #         pred_ans = vqa_model(img_ind, question)
                
        #         val_loss = criterion(pred_ans, ans)

        #         avg_loss += val_loss
            
        #     print('val_loss: %.4f' %(avg_loss/len(val_iter)))
 

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train', default='vqa_train.csv', help='input json file')
    parser.add_argument('--input_test', default='vqa_test.csv', help='input json file')
    parser.add_argument('--mapping_file', default='image_index.pkl', help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument('--image_embeddings', default='./data/img_embedding.pkl', help='output pkl file with img features')

    parser.add_argument(
        '--dataroot', default='./data/', help='path to dataset')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument(
        '--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument(
        '--imageSize',
        type=int,
        default=224,
        help='the height / width of the input image to network')
    parser.add_argument(
        '--txt_emb_size',
        type=int,
        default=100,
        help='the size of the text embedding vector')
    parser.add_argument(
        '--img_feature_size',
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
