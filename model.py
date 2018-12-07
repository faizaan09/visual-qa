import pickle as pkl
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        with open(params['image_embeddings'],'rb') as f:
            img_embs = pkl.load(f)['image_features']
        
        # self.img_features = nn.Embedding(img_embs.shape[0], img_embs.shape[1])
        # self.img_features.weight.data.copy_(img_embs)
        self.img_features = nn.Embedding.from_pretrained(torch.FloatTensor(img_embs))        
        self.text_embedding = nn.Embedding.from_pretrained(params['vocab'].vectors)

        self.parse_quest = nn.LSTM(params['txt_emb_size'], params['txt_emb_size'])
        self.hidden = self.init_lstm_hidden(params)

        self.classifier = nn.Sequential(
            nn.Linear(params['img_feature_size'] + params['txt_emb_size'], 2500),
            nn.ReLU(True),
            nn.Linear(2500, 1000), #nums_ans = 1000
            nn.Softmax(True)
        )


    def init_lstm_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, params['batch_size'], params['txt_emb_size']),
                torch.zeros(1, params['batch_size'], params['txt_emb_size']))

    def forward(self, img, quest):
        img_embedding = self.img_features(img)
        token_embeddings  = self.text_embedding(quest)
        token_embeddings = torch.cat(token_embeddings).view(token_embeddings.shape[0], 1, -1)

        quest_embedding, self.hidden = self.parse_quest(token_embeddings, self.hidden)

        quest_img_vector = torch.cat((img_embedding,quest_embedding), 0)

        answer = self.classifier(quest_img_vector)

        return answer

class Encoder(nn.Module):
    def __init__(self, img_embed, txt_embed, params):
        super(Encoder, self).__init__()
        
        self.img_features = img_embed       
        self.text_embedding = txt_embed

        self.parse_quest = nn.LSTM(params['txt_emb_size'], params['txt_emb_size'], batch_first=True)
        self.hidden = self.init_lstm_hidden(params)

        self.fusion = nn.Sequential(
            nn.Linear(params['img_feature_size'] + params['txt_emb_size'], 2500),
            nn.ReLU(True),
            nn.Linear(2500, params['txt_emb_size']),
            nn.ReLU(True)
        )


    def init_lstm_hidden(self, params):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, params['batch_size'], params['txt_emb_size']),
                torch.zeros(1, params['batch_size'], params['txt_emb_size']))

    def forward(self, img, quest):
        # batch_size = quest.shape[0]

        img_embedding = self.img_features(img)
        token_embeddings  = self.text_embedding(quest)
        # token_embeddings = torch.cat(token_embeddings).view(1, 1, -1)

        output, self.hidden = self.parse_quest(token_embeddings, self.hidden)

        quest_embedding = self.hidden[0][0]
        quest_img_vector = torch.cat((img_embedding, quest_embedding), 1)
        context = self.fusion(quest_img_vector)

        return context

class Decoder(nn.Module):
    def __init__(self, txt_embed, params):
        super(Decoder, self).__init__()
        
        self.text_embedding = txt_embed
        self.LSTM = nn.LSTM(params['txt_emb_size'], params['txt_emb_size'])
        # self.hidden = encoder_output

    def forward(self, input, hidden):
        token_embeddings  = self.text_embedding(input)

        next_word_embed, hidden = self.LSTM(token_embeddings, hidden)

        return next_word_embed, hidden


