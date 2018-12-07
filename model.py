import pickle as pkl
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, params):

        self.img_features = nn.Embedding(params['img_count'],params['img_emb_size'])

        with open(params['image_embeddings'],'rb') as f:
            img_embs = pkl.load(f)
        
        self.img_features.weight.data.copy_(img_embs)
        # self.img_features = nn.Embedding.from_pretrained(img_embs)        
        self.text_embedding = nn.Embedding.from_pretrained(params['vocab'].vectors)

        self.parse_quest = nn.LSTM(params['txt_emb_size'], params['txt_emb_size'])
        self.hidden = self.init_lstm_hidden(params)

        self.classifier = nn.Sequential(
            nn.Linear(params['img_emb_size'] + params['txt_emb_size'], 2500),
            nn.ReLU(True),
            nn.Linear(2500, params['num_ans']),
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



