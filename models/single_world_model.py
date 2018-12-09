import pickle as pkl
import torch
import torch.nn as nn
from torch.autograd import Variable


class VQAModel(nn.Module):
    def __init__(self, img_embed, txt_embed, params):
        super(Encoder_attn, self).__init__()

        self.img_features = img_embed
        self.text_embedding = txt_embed

        self.parse_quest = nn.LSTM(
            params['txt_emb_size'], params['txt_emb_size'], batch_first=True)
        self.hidden = self.init_lstm_hidden(params)

        ## attention submodule 1
        self.question_attn_fc_1 = nn.Sequential(
            nn.Linear(params['txt_emb_size'], 400), nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(400), nn.Dropout(0.2))
        self.image_attn_fc_1 = nn.Sequential(
            nn.Linear(params['img_feature_size'], 400),
            nn.LeakyReLU(inplace=True), nn.BatchNorm1d(400), nn.Dropout(0.2))
        self.attention_1 = nn.Sequential(
            nn.BatchNorm1d(400), nn.Linear(400, params['img_feature_size']),
            nn.BatchNorm1d(params['img_feature_size']), nn.Softmax())

        ##

        ## attention submodule 2
        self.question_attn_fc_2 = nn.Sequential(
            nn.Linear(params['txt_emb_size'], 400), nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(400), nn.Dropout(0.2))
        self.image_attn_fc_2 = nn.Sequential(
            nn.Linear(params['img_feature_size'], 400),
            nn.LeakyReLU(inplace=True), nn.BatchNorm1d(400), nn.Dropout(0.2))
        self.attention_2 = nn.Sequential(
            nn.BatchNorm1d(400), nn.Linear(400, params['img_feature_size']),
            nn.BatchNorm1d(params['img_feature_size']), nn.Softmax())

        ##

        self.quest_fc = nn.Sequential(
            nn.Linear(params['txt_emb_size'], 1000), nn.BatchNorm1d(1000),
            nn.Dropout(0.3))
        self.image_fc = nn.Sequential(
            nn.Linear(params['img_feature_size'], 1000), nn.BatchNorm1d(1000),
            nn.Dropout(0.3))
        self.fusion = nn.Sequential(
            nn.Linear(1000, 2500), nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(2500), nn.Dropout(0.4),
            nn.Linear(2500, params['txt_emb_size']))

    def init_lstm_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, params['batch_size'], params['txt_emb_size']),
                torch.zeros(1, params['batch_size'], params['txt_emb_size']))

    def init_hidden(self, params):
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
        # batch_size = quest.shape[0]

        img_embedding = self.img_features(img)
        token_embeddings = self.text_embedding(quest)
        # token_embeddings = torch.cat(token_embeddings).view(1, 1, -1)

        output, self.hidden = self.parse_quest(token_embeddings, self.hidden)

        quest_embedding = self.hidden[0][0]

        ## attention submodule 1
        quest_feats_1 = self.question_attn_fc_1(quest_embedding)
        img_feats_1 = self.image_attn_fc_1(img_embedding)
        attention_weights_1 = self.attention_1(
            torch.mul(quest_feats_1, img_feats_1))
        ##

        ## attention submodule 2
        quest_feats_2 = self.question_attn_fc_2(quest_embedding)
        img_feats_2 = self.image_attn_fc_2(img_embedding)
        attention_weights_2 = self.attention_2(
            torch.mul(quest_feats_1, img_feats_1))
        ##

        img_embedding = torch.mul(
            attention_weights_1 + attention_weights_2,
            img_embedding)  #attention weighted img_embedding
        ##

        ### forming the context vector
        quest_embedding = self.quest_fc(quest_embedding)
        img_embedding = self.image_fc(img_embedding)
        quest_img_vector = torch.mul(img_embedding, quest_embedding)
        # quest_img_vector = torch.cat((img_embedding, quest_embedding), 1)
        ans_embed = self.fusion(quest_img_vector)

        return ans_embed