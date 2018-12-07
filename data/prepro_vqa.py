from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import json
import pickle as pkl

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top answer and their counts:')    
    print ('\n'.join(map(str,cw[:20])))
    
    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if img['ans'] in atoi:
            new_imgs.append(img)

    print( 'question number reduce from %d to %d '%(len(imgs), len(new_imgs)))
    return new_imgs

def encode_answer(imgs, atoi):

    for i, img in enumerate(imgs):
        img['ans'] = atoi.get(img['ans'])

    return imgs

def image_to_index(imgs_train, imgs_test):
    img_to_index = {}
    index_to_img = []
    ind = 0

    for img in imgs_train:
        if img['img_path'] not in img_to_index:
            img_to_index[img['img_path']] = ind
            index_to_img.append(img['img_path'])
            img['img_path'] = ind
            ind+=1
        else:
            img['img_path'] = img_to_index[img['img_path']]

    for img in imgs_test:
        if img['img_path'] not in img_to_index:
            img_to_index[img['img_path']] = ind
            index_to_img.append(img['img_path'])
            img['img_path'] = ind
            ind+=1
        else:
            img['img_path'] = img_to_index[img['img_path']]
    
    image_index = (img_to_index, index_to_img)

    with open('image_index.pkl','wb') as f:
        pkl.dump(image_index,f)

    return imgs_train, imgs_test

def main(params):

    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    #imgs_train = imgs_train[:5000]
    #imgs_test = imgs_test[:5000]
    # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w:i for i,w in enumerate(top_ans)}
    itoa = {i:w for i,w in enumerate(top_ans)}

    # filter question, which isn't in the top answers.
    imgs_train = filter_question(imgs_train, atoi)
    imgs_test = filter_question(imgs_test, atoi)

    imgs_train = encode_answer(imgs_train,atoi)

    imgs_test = encode_answer(imgs_test,atoi)


    ## remove unnecessary field from data
    for data in imgs_train:
        data.pop('MC_ans', None)

    for data in imgs_test:
        data.pop('MC_ans', None)

    imgs_train, imgs_test = image_to_index(imgs_train, imgs_test)

    json.dump(imgs_train, open(params['output_train_json'], 'w'))
    json.dump(imgs_test, open(params['output_test_json'], 'w'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='../data/vqa_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='../data/vqa_raw_test.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    parser.add_argument('--output_train_json', default='../data/vqa_base_train.json', help='output json file')
    parser.add_argument('--output_test_json', default='../data/vqa_base_test.json', help='output json file')
    
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))
    main(params)
