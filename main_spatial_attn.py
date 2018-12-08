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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from model import Encoder_attn, Decoder
from metrics import filterOutput, maskedLoss, word_accuracy
from torchtext.data import TabularDataset, Field, Iterator
from tensorboardX import SummaryWriter

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

    SOS_token = '<sos>'
    EOS_token = '<eos>'
    PAD_token = '<pad>'

    TEXT = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
        init_token=SOS_token,
        eos_token=EOS_token)
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

    PAD_token_ind = vocab.stoi[PAD_token]
    SOS_token_ind = vocab.stoi[SOS_token]
    EOS_token_ind = vocab.stoi[EOS_token]

    print("Creating Embedding from vocab vectors ..")
    txt_embed = nn.Embedding.from_pretrained(vocab.vectors)
    print("Text Embeddings are generated of size ", txt_embed.weight.size())

    print("Loading Image embeddings ..")
    with open(params['image_embeddings'], 'rb') as f:
        img_embs = pkl.load(f)['image_features']

    img_embed = nn.Embedding.from_pretrained(torch.FloatTensor(img_embs))

    print("Creating Encoder_attn ..")
    encoder = Encoder_attn(img_embed, txt_embed, params)
    print(encoder)

    print("Creating Decoder ..")
    decoder = Decoder(txt_embed, params)
    print(decoder)

    criterion = torch.nn.PairwiseDistance(keepdim=False)
    criterion.to(device)

    ## [Completed] TODO(Jay) : Remove this check and use .to(device)
    # if params['cuda']:
    #     encoder.cuda()
    #     decoder.cuda()
    #     criterion.cuda()

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=params['lr'], weight_decay=1e-5, amsgrad=True)
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(), lr=params['lr'], weight_decay=1e-5, amsgrad=True)

    encoder_LR_scheduler = ReduceLROnPlateau(
        encoder_optimizer, 'min', patience=1)
    decoder_LR_scheduler = ReduceLROnPlateau(
        decoder_optimizer, 'min', patience=1)

    if params['use_checkpoint']:
        checkpoint = torch.load(params['enc_dec_model'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        encoder_optimizer.load_state_dict(
            checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(
            checkpoint['decoder_optimizer_state_dict'])
        encoder_LR_scheduler.load_state_dict(checkpoint['encoder_LR_scheduler'])
        decoder_LR_scheduler.load_state_dict(checkpoint['decoder_LR_scheduler'])

    encoder.to(device)
    decoder.to(device)

    train_iter, val_iter = Iterator.splits((train, val),
                                           batch_sizes=(params['batch_size'],
                                                        params['batch_size']),
                                           sort=False,
                                           shuffle=True,
                                           device=device)

    for epoch in range(params['niter']):

        for is_train in (True, False):
            print('Is Training: ', is_train)
            if is_train:
                encoder.train()
                decoder.train()
                data_iter = train_iter
            else:
                encoder.eval()
                decoder.eval()
                data_iter = val_iter

            total_loss = 0
            total_acc = 0

            with torch.set_grad_enabled(is_train):

                for i, row in enumerate(data_iter, 1):

                    if len(row) < params['batch_size']:
                        continue

                    encoder.zero_grad()
                    decoder.zero_grad()

                    ans, img_ind, question = row.ans, row.img_ind, row.question
                    batch_size = params['batch_size']

                    ## target_length-1 since we are not predicting SOS token
                    target_length = ans.shape[1] - 1

                    encoder.hidden = encoder.init_hidden(params)

                    ans = ans.to(device)
                    img_ind = img_ind.to(device)
                    question = question.to(device)
                    encoder.hidden = (encoder.hidden[0].to(device),
                                      encoder.hidden[1].to(device))

                    ans_embed = txt_embed(ans)

                    encoder_output = encoder(img_ind, question)

                    decoder_input = ans_embed[:, 0].reshape(
                        (batch_size, 1, -1))  ## (batch_size, 1) check again
                    ans_embed = ans_embed[:, 1:]  ## removed the SOS token
                    ans = ans[:, 1:]  ## removed the SOS token

                    decoder_hidden = decoder.init_hidden(encoder_output, params)

                    if params['cuda']:
                        decoder_hidden = (decoder_hidden[0].cuda(),
                                          decoder_hidden[1].cuda())

                    outputs = torch.zeros(batch_size, target_length,
                                          params['txt_emb_size'])

                    ## [Completed] TODO(Jay) : remove the sos token from the ans and ans_embed before calc loss and acc
                    for di in range(target_length - 1):
                        decoder_output, decoder_hidden = decoder(
                            decoder_input, decoder_hidden)

                        ## TODO(Jay) : Detach the input from history
                        decoder_input = decoder_output

                        outputs[:, di, :] = decoder_output.reshape(
                            batch_size, -1)

                    filtered_labels, filtered_label_embeds, filtered_outputs = filterOutput(
                        outputs.reshape(batch_size * target_length, -1),
                        ans.reshape(batch_size * target_length, -1),
                        ans_embed.reshape(batch_size * target_length, -1),
                        PAD_token_ind)

                    filtered_label_embeds = filtered_label_embeds.to(device)
                    filtered_outputs = filtered_outputs.to(device)

                    batch_loss = maskedLoss(filtered_label_embeds,
                                            filtered_outputs, criterion)

                    batch_acc = word_accuracy(filtered_outputs,
                                              vocab.vectors.to(device),
                                              filtered_labels)

                    total_loss += batch_loss.item()
                    total_acc += batch_acc

                    if is_train:
                        if i % 1000 == 0:
                            print(
                                '[%d/%d][%d/%d] train_loss: %.4f, Accuracy: %.4f'
                                % (epoch, params['niter'], i, len(data_iter),
                                   total_loss / i, total_acc / i))

                        batch_loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()

                avg_loss = total_loss / len(data_iter)
                avg_acc = total_acc / len(data_iter)

                if is_train:
                    PATH = os.path.join(output_dir, 'enc_dec_model.pth')
                    torch.save({
                        'encoder_state_dict':
                        encoder.state_dict(),
                        'decoder_state_dict':
                        decoder.state_dict(),
                        'encoder_optimizer_state_dict':
                        encoder_optimizer.state_dict(),
                        'decoder_optimizer_state_dict':
                        decoder_optimizer.state_dict(),
                        'encoder_LR_scheduler':
                        encoder_LR_scheduler.state_dict(),
                        'decoder_LR_scheduler':
                        decoder_LR_scheduler.state_dict(),
                    }, PATH)

                    writer.add_scalars('data', {
                        'train_loss': avg_loss,
                        'train_acc': avg_acc
                    }, epoch)
                else:
                    print('Calculating Validation loss')
                    print(
                        'val_loss: %.4f, Accuracy: %.4f' % (avg_loss, avg_acc))

                    encoder_LR_scheduler.step(avg_loss)
                    decoder_LR_scheduler.step(avg_loss)

                    writer.add_scalars('data', {
                        'val_loss': avg_loss,
                        'val_acc': avg_acc
                    }, epoch)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument(
        '--input_train', default='vqa_train.csv', help='input json file')
    parser.add_argument(
        '--input_test', default='vqa_test.csv', help='input json file')
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
        help='Flag which states whether to use the previous Model checkpoint')
    parser.add_argument(
        '--enc_dec_model',
        default='output/enc_dec_model.pth',
        help='Saved model path')
    parser.add_argument('--dataroot', default='./data/', help='path to dataset')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument(
        '--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--n_layers', type=int, default=2, help='Num of layers in LSTM')
    parser.add_argument('--bidirection', help='Bidirectional LSTM')
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