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


class Encoder(nn.Module):

    def __init__(self, img_embed, txt_embed, params):
        super(Encoder, self).__init__()

        self.img_features = img_embed
        self.text_embedding = txt_embed
        self.n_layers = params['n_layers']
        self.direction = 1 + int(params['bidirection'])

        ## TODO(Jay) : Change the model output size for bidirectional
        self.parse_quest = nn.LSTM(
            params['txt_emb_size'],
            params['txt_emb_size'],
            num_layers=self.n_layers,
            dropout=0.3,
            bidirectional=params['bidirection'],
            batch_first=True)

        self.hidden = self.init_hidden(params)

        self.fusion = nn.Sequential(
            nn.BatchNorm1d(params['img_feature_size'] + params['txt_emb_size']),
            nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(params['img_feature_size'] + params['txt_emb_size'],
                      2500), nn.BatchNorm1d(2500), nn.LeakyReLU(True),
            nn.Dropout(), nn.Linear(2500, params['txt_emb_size']),
            nn.LeakyReLU(True))

    def init_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(params['n_layers'] * self.direction,
                            params['batch_size'], params['txt_emb_size']),
                torch.zeros(params['n_layers'] * self.direction,
                            params['batch_size'], params['txt_emb_size']))

    def forward(self, img, quest):
        # batch_size = quest.shape[0]

        img_embedding = self.img_features(img)
        token_embeddings = self.text_embedding(quest)
        # token_embeddings = torch.cat(token_embeddings).view(1, 1, -1)

        output, self.hidden = self.parse_quest(token_embeddings, self.hidden)

        quest_embedding = self.hidden[0][0]
        quest_img_vector = torch.cat((img_embedding, quest_embedding), 1)
        context = self.fusion(quest_img_vector)

        return context


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
        self.cnn.eval()
        self.output_size = (-1, 512, 7, 7)

        for param in self.cnn.parameters():
            param.requires_grad = False

        # self.fc = nn.Sequential(nn.Linear(512, output_size), nn.Tanh())

    def forward(self, image):
        # N * 224 * 224 -> N * 512 * 7 * 7
        image_features = self.cnn(image)

        return image_features


class Encoder_2d_attn(nn.Module):

    def __init__(self, img, txt_embed, params):
        super(Encoder_2d_attn, self).__init__()

        self.n_layers = params['n_layers']
        self.direction = 1 + int(params['bidirection'])

        self.img_features = ImageEmbedding()
        self.text_embedding = txt_embed
        self.feature_map_size = self.img_features.output_size

        attention_input_size = feature_map_side**2 + params['txt_emb_size']

        ## TODO(Jay) : Change the model output size for bidirectional
        self.parse_quest = nn.LSTM(
            params['txt_emb_size'],
            params['txt_emb_size'],
            num_layers=self.n_layers,
            dropout=0.3,
            bidirectional=params['bidirection'],
            batch_first=True)

        self.hidden = self.init_hidden(params)

        self.attention = nn.Sequential(
            nn.Linear(attention_input_size, 49), nn.Sigmoid())

        self.fusion = nn.Sequential(
            nn.BatchNorm1d(params['img_feature_size'] + params['txt_emb_size']),
            nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(params['img_feature_size'] + params['txt_emb_size'],
                      2500), nn.BatchNorm1d(2500), nn.LeakyReLU(True),
            nn.Dropout(), nn.Linear(2500, params['txt_emb_size']),
            nn.LeakyReLU(True))

    def init_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(params['n_layers'] * self.direction,
                            params['batch_size'], params['txt_emb_size']),
                torch.zeros(params['n_layers'] * self.direction,
                            params['batch_size'], params['txt_emb_size']))

    def forward(self, img, quest):
        batch_size = quest.shape[0]

        img_embedding = self.img_features(img)  ## batch x 512 x 7 x 7
        token_embeddings = self.text_embedding(quest)  ## batch x sent_len x 100
        ## TODO(Jay) : Add dropout after text embeddings

        output, self.hidden = self.parse_quest(token_embeddings, self.hidden)
        quest_embedding = self.hidden[0][0]  ## batch x 100

        ## convert img_embed to ((batch x 512) x 49)
        img_embedding_feats = img_embedding.reshape(
            batch_size * self.feature_map_size[1], -1)

        ## convert quest_embed (batch x 100) to ((batch x 512) x 49)
        quest_embedding_feats = self.question_attn_fc(100, 49)  ## batch x 49
        # quest_embedding_feats = quest_embedding_feats.reshape(
        #     batch_size, 1, -1)  ## batch x 1 x 49
        quest_embedding_feats = quest_embedding_feats.repeat(
            512, 1)  ## (batch x 512) x 49

        ## multiply the quest_embedding_feats and img_embedding and pass
        # it to the attn layer to generate the attention weights
        self.attention(quest_embedding_feats.mul(img_embedding_feats))

        ## attention_weights size (batch x 512 x 7 x 7)

        ## update -> img_embed * attention_weights
        ## sum over the channel dimension, output -> batch x 1 x 7 x 7

        quest_img_vector = torch.cat((img_embedding, quest_embedding), 1)
        context = self.fusion(quest_img_vector)

        return context
