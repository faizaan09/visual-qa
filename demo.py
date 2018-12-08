from tkinter.filedialog import askopenfilename
import torch
import model
import os
import pickle as pkl
import spacy
import argparse
import json
from torch.autograd import Variable
from tkinter import Tk



def main(params):

    if params['cuda']:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    with open(params['text_embeddings'],'rb') as f:
        vocab = pkl.load(f)
        params['vocab'] = vocab

    
    print("Select Model for evaluation:")
    print("1. Choose answer from predefined set")
    print("2. Generate new answer")
    choice = input()
    
    if choice == "1":

        vqa_model = model.Model(params)

        checkpoint = torch.load(params['baseline_model'])
        vqa_model.load_state_dict(checkpoint['model_state_dict'])
        vqa_model.hidden = checkpoint['lstm_hidden']
        
        vqa_model.cuda()
        vqa_model.hidden = tuple([v.cuda() for v in vqa_model.hidden])
        vqa_model.eval()

    else:
        pass

    while True:

        print("Choose an image for the demo: \n")
        
        dir = os.path.join('D:','Documents','Fall 2018','visual-qa','data','train2014')
        Tk().withdraw()
        filename = askopenfilename(initialdir = dir, title = 'Select image')
        filename = filename[filename.find('train'):]
        
        quest = input("Please enter the question: \n")

        img, quest = preprocess(filename, quest, params)

        img = torch.LongTensor([img])
        quest = torch.LongTensor(quest)

        test_img = torch.zeros((params['batch_size'], 1), dtype=torch.int64)
        test_quest = torch.zeros((params['batch_size'], len(quest)), dtype=torch.int64)

        test_img[0] = img
        test_img = test_img.squeeze()
        test_quest[0] = quest
        
        if params['cuda']:
            test_img = test_img.cuda()
            test_quest = test_quest.cuda()
            # vqa_model.cuda()
            
        
        test_img = Variable(test_img)
        test_quest = Variable(test_quest)
        ans = vqa_model(test_img, test_quest)
        
        with open(params['answer_mapping_file'],'rb') as f:
            atoi, itoa = pkl.load(f)
        
        possible_answers = [itoa[ind.item()] for ind in ans.topk(5)[1][0]]
        print("Possible answers are: \n")
        for i in range(5):
            print(possible_answers[i])

        flag = input("enter q to exit, any other key to continue\n")
        if flag == "q":
            break


    

    


spacy_en = spacy.load('en')
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

def preprocess(img_path, quest, params):
    with open(params['mapping_file'],'rb') as f:
        img2ind = pkl.load(f)

    img_ind = img2ind[0][img_path]

    quest = quest.lower()
    quest = tokenizer(quest)

    quest = [params['vocab'].stoi[word] for word in quest]

    return img_ind, quest

    




    
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument('--baseline_model', default='output/20181208_0119/baseline_15.pth', help='saved baseline model path')
    parser.add_argument('--lstm_model', default='20181207_1945/blah.pkl', help='saved LSTM model path')
    parser.add_argument('--mapping_file', default='data/image_index.pkl', help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument('--answer_mapping_file', default='data/answer_index.pkl', help='This files contains the ans_id to word mapping and vice versa')
    parser.add_argument('--image_embeddings', default='./data/img_embedding.pkl', help='output pkl file with img features')
    parser.add_argument('--text_embeddings', default='./data/vocab.pkl', help='output pkl file with text embeddings')

    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument(
        '--outf',
        default='./output/',
        help='folder to output images and model checkpoints')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--txt_emb_size',
        type=int,
        default=100,
        help='the size of the text embedding vector')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument(
        '--eval', 
        action='store_true', 
        help="choose whether to train the model or show demo")
    parser.add_argument(
        '--img_feature_size',
        type=int,
        default=2048,
        help='the size of the image feature vector')
    
    
    args = parser.parse_args()
    params = vars(args)

    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))
    
    main(params)