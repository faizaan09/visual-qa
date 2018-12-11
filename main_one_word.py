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
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from datetime import datetime
from models.single_world_model import VQAModel
from metrics import filterOutput, maskedLoss, word_accuracy, getIndicesFromEmbedding
from torchtext.data import TabularDataset, Field, Iterator
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

spacy_en = spacy.load('en')


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def main(params):
    try:
        output_dir = os.path.join(
            params['outf'], datetime.strftime(datetime.now(), "%Y%m%d_%H%M"))
        os.makedirs(output_dir)
    except OSError:
        pass

    if torch.cuda.is_available() and not params['cuda']:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    writer = SummaryWriter(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEXT = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True)
    # LABEL = Field(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True, batch_first=True, init_token='#', eos_token='$')
    IMG_IND = Field(sequential=False, use_vocab=False, batch_first=True)

    fields = {
        'ans': ('ans', TEXT),
        'img_ind': ('img_ind', IMG_IND),
        'question': ('question', TEXT)
    }

    train, val = TabularDataset.splits(
        path=params['dataroot'],
        train=params['input_train'],
        validation=params['input_test'],
        format='csv',
        skip_header=False,
        fields=fields,
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

    with open('data/one_word_vocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)

    print("Creating Embedding from vocab vectors ..")
    txt_embed = nn.Embedding.from_pretrained(vocab.vectors)
    print("Text Embeddings are generated of size ", txt_embed.weight.size())

    print("Loading Image embeddings ..")
    with open(params['image_embeddings'], 'rb') as f:
        img_embs = pkl.load(f)['image_features']

    img_embed = nn.Embedding.from_pretrained(torch.FloatTensor(img_embs))

    print("Creating VQAModel ..")
    word_gen_model = VQAModel(img_embed, txt_embed, params)
    print(word_gen_model)

    y = torch.FloatTensor([1]).to(device)
    criterion = torch.nn.CosineEmbeddingLoss()
    criterion.to(device)
    word_gen_model.to(device)

    optimizer = torch.optim.Adam(
        word_gen_model.parameters(),
        lr=params['lr'],
        weight_decay=1e-5,
        amsgrad=True)

    LR_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

    # optimizer = torch.optim.Adagrad(
    #     word_gen_model.parameters(), lr=params['lr'], weight_decay=1e-5)

    if params['use_checkpoint']:
        checkpoint = torch.load(params['enc_dec_model'])
        word_gen_model.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        LR_scheduler.load_state_dict(checkpoint['LR_scheduler'])

    for epoch in range(params['niter']):

        train_iter, val_iter = Iterator.splits(
            (train, val),
            batch_sizes=(params['batch_size'], params['batch_size']),
            sort=False,
            shuffle=True,
            device=device)

        for is_train in (True, False):
            print('Is Training: ', is_train)
            if is_train:
                word_gen_model.train()
                data_iter = train_iter
            else:
                word_gen_model.eval()
                data_iter = val_iter

            total_loss = 0
            total_acc = 0

            with torch.set_grad_enabled(is_train):

                for i, row in enumerate(data_iter, 1):

                    if len(row) < params['batch_size']:
                        continue

                    word_gen_model.zero_grad()

                    ans, img_ind, question = row.ans, row.img_ind, row.question

                    # to skip examples where we have a 1 word ans, but after tokenization it becomes 2 words
                    # e.g. mcdonald's -> [mcdonald,'s]
                    if ans.shape[1] != 1:
                        continue

                    word_gen_model.hidden = word_gen_model.init_hidden(params)

                    ans.squeeze_()
                    ans = ans.to(device)
                    img_ind = img_ind.to(device)
                    question = question.to(device)
                    word_gen_model.hidden = (
                        word_gen_model.hidden[0].to(device),
                        word_gen_model.hidden[1].to(device))

                    img_ind = Variable(img_ind)
                    question = Variable(question)

                    encoder_output = word_gen_model(img_ind, question)

                    ans_embed = txt_embed(ans)
                    batch_loss = criterion(encoder_output, ans_embed, y)

                    batch_acc = word_accuracy(encoder_output,
                                              vocab.vectors.to(device), ans)

                    total_loss += batch_loss.item()
                    total_acc += batch_acc.item()

                    if is_train:
                        if i % 1000 == 0:
                            print(
                                '[%d/%d][%d/%d] train_loss: %.4f, Accuracy: %.4f'
                                % (epoch, params['niter'], i, len(data_iter),
                                   total_loss / i, total_acc / i))

                        batch_loss.backward()
                        optimizer.step()

                avg_loss = total_loss / len(data_iter)
                avg_acc = total_acc / len(data_iter)

                if is_train:
                    PATH = os.path.join(output_dir, 'enc_dec_model.pth')
                    torch.save(
                        {
                            'encoder_state_dict': word_gen_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'LR_scheduler': LR_scheduler.state_dict(),
                        }, PATH)

                    writer.add_scalars('data', {
                        'train_loss': avg_loss,
                        'train_acc': avg_acc
                    }, epoch)
                else:
                    print('Calculating Validation loss')
                    print(
                        'val_loss: %.4f, Accuracy: %.4f' % (avg_loss, avg_acc))

                    LR_scheduler.step(avg_loss)

                    writer.add_scalars('data', {
                        'val_loss': avg_loss,
                        'val_acc': avg_acc
                    }, epoch)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument(
        '--input_train',
        default='vqa_one_word_train.csv',
        help='input csv file')
    parser.add_argument(
        '--input_test', default='vqa_one_word_test.csv', help='input csv file')
    parser.add_argument(
        '--mapping_file',
        default='image_index.pkl',
        help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument(
        '--image_embeddings',
        default='./data/img_embedding.pkl',
        help='output pkl file with img features')
    parser.add_argument(
        '--use_checkpoint',
        action='store_true',
        help='Flag which states whether to use the previous Model checkpoint')
    parser.add_argument(
        '--enc_dec_model',
        default='output/20181208_0927/enc_dec_model.pth',
        help='Saved model path')
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
        '--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument(
        '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument(
        '--cuda', default=True, action='store_true', help='enables cuda')
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