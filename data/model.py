import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, params):

        self.img_features = nn.Embedding(params['img_count'],params['img_emb_size'])

        self.text_embedding = nn.Embedding(len(params['vocab']), params['txt_emb_size'])
        self.text_embedding.weight.data.copy_(params['vocab'].vectors)

        self.parse_quest = nn.Sequential(
            nn.LSTM()
        )
        self.classifier = nn.Sequential(
            nn.Linear(params['img_emb_size'] + params['txt_emb_size'], 2500)
            nn.ReLU(True)
            nn.Linear(2500, params['num_ans'])
            nn.Softmax(True)
        )

    def forward(self,img_ind,quest):
        encoded_img = self.img_features(img)

