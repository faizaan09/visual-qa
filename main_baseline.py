from __future__ import print_function
import os
import torch
import random
import argparse
import json
import spacy
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
import model
from torchtext.data import TabularDataset, Field, Iterator
import pickle as pkl
from tensorboardX import SummaryWriter
spacy_en = spacy.load('en')


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def repackage_hidden(hidden):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if not type(hidden) == tuple:
        return Variable(hidden)
    else:
        return tuple(repackage_hidden(variable) for variable in hidden)


def main(params):
    try:
        output_dir = os.path.join(
            params['outf'], datetime.strftime(datetime.now(), "%Y%m%d_%H%M"))
        os.makedirs(output_dir)
    except OSError:
        pass

    writer = SummaryWriter(output_dir)
    if torch.cuda.is_available() and not params['cuda']:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    TEXT = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True)
    LABEL = Field(
        sequential=False, use_vocab=False, is_target=True, batch_first=True)
    IMG_IND = Field(sequential=False, use_vocab=False, batch_first=True)

    fields = {
        'ans': ('ans', LABEL),
        'img_ind': ('img_ind', IMG_IND),
        'question': ('question', TEXT)
    }

    train, val = TabularDataset.splits(
        path=params['dataroot'],
        train=params['input_train'],
        validation=params['input_test'],
        format='csv',
        skip_header=False,
        fields=fields)

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
    params['vocab'] = vocab

    vqa_model = model.Model(params)

    print(vqa_model)

    if params['use_checkpoint']:
        checkpoint = torch.load(params['mcq_model'])
        vqa_model.load_state_dict(checkpoint['model_state_dict'])
        vqa_model.hidden = checkpoint['lstm_hidden']

    criterion = torch.nn.CrossEntropyLoss()

    if params['cuda']:
        vqa_model.cuda()
        criterion.cuda()

    optimizer = torch.optim.Adam(vqa_model.parameters(), lr=params['lr'])

    train_iter, val_iter = Iterator.splits((train, val),
                                           batch_sizes=(params['batch_size'],
                                                        params['batch_size']),
                                           sort_within_batch=False,
                                           sort=False)

    for epoch in range(1, params['niter'] + 1):

        total_val_loss = 0
        total_val_matches = 0
        total_train_loss = 0
        total_train_matches = 0

        for i, row in enumerate(train_iter):

            vqa_model.train()
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            if len(row) < params['batch_size']:
                continue
            vqa_model.hidden = repackage_hidden(vqa_model.hidden)
            vqa_model.zero_grad()
            ans, img_ind, question = row.ans, row.img_ind, row.question

            batch_size = ans.size(0)

            if params['cuda']:
                ans = ans.cuda()
                img_ind = img_ind.cuda()
                question = question.cuda()
                vqa_model.hidden = tuple([v.cuda() for v in vqa_model.hidden])

            ans_var = Variable(ans)
            img_ind_var = Variable(img_ind)
            question_var = Variable(question)

            pred_ans = vqa_model(img_ind_var, question_var)

            train_loss = criterion(pred_ans, ans_var)

            pred_ind = pred_ans.max(dim=1)[1]
            train_acc = (pred_ind == ans_var).sum()

            total_train_loss += train_loss.item()
            total_train_matches += train_acc.item()

            train_loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('[%d/%d][%d/%d] train_loss: %.4f' %
                      (epoch, params['niter'], i + 1, len(train_iter),
                       train_loss))

        vqa_model.eval()
        for row in val_iter:

            if len(row) < params['batch_size']:
                continue

            vqa_model.hidden = repackage_hidden(vqa_model.hidden)
            vqa_model.zero_grad()
            ans, img_ind, question = row.ans, row.img_ind, row.question

            batch_size = ans.size(0)

            if params['cuda']:
                ans = ans.cuda()
                img_ind = img_ind.cuda()
                question = question.cuda()
                vqa_model.hidden = tuple([v.cuda() for v in vqa_model.hidden])

            ans_var = Variable(ans)
            img_ind_var = Variable(img_ind)
            question_var = Variable(question)

            pred_ans = vqa_model(img_ind_var, question_var)

            val_loss = criterion(pred_ans, ans_var)
            pred_ind = pred_ans.max(dim=1)[1]
            val_acc = (pred_ind == ans_var).sum()
            total_val_loss += val_loss.item()
            total_val_matches += val_acc.item()

        print(
            '[%d/%d] train_loss: %.4f val_loss: %.4f train_acc: %.4f val_acc: %.4f'
            % (epoch, params['niter'], total_train_loss / len(train_iter),
               total_val_loss / len(val_iter), total_train_matches * 100 /
               len(train_iter) / params['batch_size'],
               total_val_matches * 100 / len(val_iter) / params['batch_size']))

        writer.add_scalars(
            'data', {
                'train_loss':
                train_loss,
                'train_acc':
                total_train_matches * 100 / len(train_iter) /
                params['batch_size'],
                'val_loss':
                total_val_loss / len(val_iter),
                'val_acc':
                total_val_matches * 100 / len(val_iter) / params['batch_size']
            }, epoch)

        torch.save({
            'lstm_hidden': vqa_model.hidden,
            'model_state_dict': vqa_model.state_dict()
        }, '%s/baseline_%d.pth' % (output_dir, epoch))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument(
        '--input_train', default='vqa_base_train.csv', help='input json file')
    parser.add_argument(
        '--input_test', default='vqa_base_test.csv', help='input json file')
    parser.add_argument(
        '--mapping_file',
        default='image_index.pkl',
        help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument(
        '--image_embeddings',
        default='./data/img_embedding.pkl',
        help='output pkl file with img features')
    parser.add_argument(
        '--mcq_model',
        default='output/checkpoint/baseline_15.pth',
        help='saved baseline model path')
    parser.add_argument(
        '--dataroot', default='./data/', help='path to dataset')
    parser.add_argument(
        '--workers',
        type=int,
        help='number of data loading workers',
        default=2)
    parser.add_argument(
        '--batch_size', type=int, default=32, help='input batch size')
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
        '--niter', type=int, default=15, help='number of epochs to train for')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0002,
        help='learning rate, default=0.0002')
    parser.add_argument(
        '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument(
        '--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument(
        '--use_checkpoint',
        action='store_true',
        help='loads saved model from "mcq_model"',
        default=True)
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

    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    main(params)
