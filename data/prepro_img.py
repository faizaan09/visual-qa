import argparse
from PIL import Image
import json
import numpy as np
import pickle as pkl
import torch
from torch.autograd import Variable
from torchvision import models, transforms

def get_image_features(image_path, model, layer):
    # 1. Load the image with Pillow library
    img = Image.open(image_path)

    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.numpy()


def load_model(end_layer='avgpool'):
    # Load the pretrained model
    model = models.resnet50(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get(end_layer)
    model.eval()

    return model, layer


def main(params):
    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))
    
    with open(params['mapping_file'], 'rb') as map_file:
        _, index_to_img = pkl.load(map_file)

    train_embeddings = np.zeros(shape=(len(imgs_train), 1000))
    test_embeddings = np.zeros(shape=(len(imgs_test), 1000))

    model, layer = load_model()

    for i, img in enumerate(imgs_train):
        img_id = img['img_id']
        features = get_image_features(index_to_img[img_id], model, layer)
        train_embeddings[i] = features

    for i, img in enumerate(imgs_test):
        img_id = img['img_id']
        features = get_image_features(index_to_img[img_id], model, layer)
        test_embeddings[i] = features

    out = {}
    out['train_embeddings'] = train_embeddings
    out['test_embeddings'] = test_embeddings

    ## save the created embeddings
    with open(params['embedding_output_path'], 'wb') as out_file:
        pkl.dump(out, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='vqa_train.json', help='input json file to process into pkl')
    parser.add_argument('--input_test_json', default='vqa_test.json', help='input json file to process into pkl')
    parser.add_argument('--mapping_file', default='image_index.pkl', help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument('--output_train_json', default='../data/vqa_train_img_feats.json', help='output json file with img features')
    parser.add_argument('--output_test_json', default='../data/vqa_test_img_feats.json', help='output json file with img features')
    parser.add_argument

    args = parser.parse_args()
    params = vars(args)

    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))
    
    main(params)
