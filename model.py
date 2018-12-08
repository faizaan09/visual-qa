import pickle as pkl
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, params):
        super(Model, self).__init__()

        with open(params['image_embeddings'], 'rb') as f:
            img_embs = pkl.load(f)['image_features']

        self.img_features = nn.Embedding(img_embs.shape[0], img_embs.shape[1])

        self.img_features.weight.data.copy_(torch.from_numpy(img_embs))
        self.img_features.weight.requires_grad = False
        # self.img_features = nn.Embedding.from_pretrained(img_embs)
        self.text_embedding = nn.Embedding.from_pretrained(
            params['vocab'].vectors)

        self.parse_quest = nn.LSTM(
            params['txt_emb_size'], params['txt_emb_size'], batch_first=True)
        self.hidden = self.init_lstm_hidden(params)

        self.classifier = nn.Sequential(
            nn.Linear(params['img_feature_size'] + params['txt_emb_size'],
                      2500),
            nn.ReLU(True),
            nn.Linear(2500, 1000)  #nums_ans = 1000
        )

    def init_lstm_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(
            torch.zeros(1, params['batch_size'], params['txt_emb_size'])),
                Variable(
                    torch.zeros(1, params['batch_size'],
                                params['txt_emb_size'])))

    def forward(self, img, quest):
        img_embedding = self.img_features(img)
        token_embeddings = self.text_embedding(quest)
        # token_embeddings = torch.cat((token_embeddings)).view(token_embeddings.shape[0], 1, -1)

        token_lstm_output, self.hidden = self.parse_quest(
            token_embeddings, self.hidden)

        quest_embeddings = self.hidden[0][0]
        quest_img_vector = torch.cat((img_embedding, quest_embeddings), 1)

        answer = self.classifier(quest_img_vector)

        return answer


class Encoder_attn(nn.Module):

    def __init__(self, img_embed, txt_embed, params):
        super(Encoder_attn, self).__init__()

        self.img_features = img_embed
        self.text_embedding = txt_embed

        self.parse_quest = nn.LSTM(
            params['txt_emb_size'], params['txt_emb_size'], batch_first=True)
        self.hidden = self.init_hidden(params)

        ## attention submodule
        self.question_attn_fc = nn.Sequential(
            nn.Linear(params['txt_emb_size'], 400), nn.LeakyReLU(inplace=True))
        self.image_attn_fc = nn.Sequential(
            nn.Linear(params['img_feature_size'], 400),
            nn.LeakyReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Linear(400, params['img_feature_size']), nn.Softmax())

        ##
        self.quest_fc = nn.Linear(params['txt_emb_size'], 1000)
        self.image_fc = nn.Linear(params['img_feature_size'], 1000)
        self.fusion = nn.Sequential(
            nn.Linear(1000, 2500), nn.LeakyReLU(inplace=True),
            nn.Linear(2500, params['txt_emb_size']), nn.LeakyReLU(inplace=True))

    def init_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, params['batch_size'], params['txt_emb_size']),
                torch.zeros(1, params['batch_size'], params['txt_emb_size']))

    def forward(self, img, quest):
        # batch_size = quest.shape[0]

        img_embedding = self.img_features(img)
        token_embeddings = self.text_embedding(quest)
        # token_embeddings = torch.cat(token_embeddings).view(1, 1, -1)

        output, self.hidden = self.parse_quest(token_embeddings, self.hidden)

        quest_embedding = self.hidden[0][0]

        ## attention submodule
        quest_feats = self.question_attn_fc(quest_embedding)
        img_feats = self.image_attn_fc(img_embedding)
        attention_weights = self.attention(torch.mul(quest_feats, img_feats))

        img_embedding = torch.mul(
            attention_weights, img_embedding)  #attention weighted img_embedding
        ##

        ### forming the context vector
        quest_embedding = self.quest_fc(quest_embedding)
        img_embedding = self.image_fc(img_embedding)
        quest_img_vector = torch.mul(img_embedding, quest_embedding)
        # quest_img_vector = torch.cat((img_embedding, quest_embedding), 1)
        context = self.fusion(quest_img_vector)

        return context


class Decoder(nn.Module):

    def __init__(self, txt_embed, params):
        super(Decoder, self).__init__()

        self.relu = torch.nn.ReLU()
        # self.text_embedding = txt_embed
        self.LSTM = nn.LSTM(
            params['txt_emb_size'], params['txt_emb_size'], batch_first=True)
        # self.hidden = encoder_output

    def init_hidden(self, encoder_output, params):
        return (encoder_output.reshape(
            shape=(1, params['batch_size'], params['txt_emb_size'])),
                torch.zeros(1, params['batch_size'], params['txt_emb_size']))

    def forward(self, input, hidden):

        # token_embeddings  = self.text_embedding(input)
        token_embeddings = input
        token_embeddings = self.relu(token_embeddings)
        next_word_embed, hidden = self.LSTM(token_embeddings, hidden)

        return next_word_embed, hidden


class ImageEmbedding(nn.Module):

    def __init__(self):  #, output_size=1024):
        super(ImageEmbedding, self).__init__()
        self.cnn = models.vgg19_bn(pretrained=True).features

        for param in self.cnn.parameters():
            param.requires_grad = False

        # self.fc = nn.Sequential(nn.Linear(512, output_size), nn.Tanh())

    def forward(self, image, image_ids):
        # N * 224 * 224 -> N * 512 * 7 * 7
        image_features = self.cnn(image)

        return image_features
