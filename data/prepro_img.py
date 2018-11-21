import argparse
from PIL import Image
import json
import numpy as np
import pickle as pkl
import torch
from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm
import sys

def get_image_features(image_paths, model, layer):
    # 1. Load the image with Pillow library

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    t_imgs = torch.zeros([len(image_paths),3, 224, 224])
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        img = img.convert(mode='RGB')
        t_imgs[i,:,:,:] = normalize(to_tensor(scaler(img)))

    # 2. Create a PyTorch Variable with the transformed image
    t_imgs = Variable(t_imgs.cuda())

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros((len(image_paths),2048))
    my_embedding = Variable(my_embedding.cuda())
    
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.squeeze().data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_imgs)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.cpu().numpy()


def load_model(end_layer='avgpool'):
    # Load the pretrained model
    model = models.resnet50(pretrained=True)

    # Use the model object to select the desired layer
    layer = model._modules.get(end_layer)
    model.eval()

    return model, layer


def main(params):    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(params['mapping_file'], 'rb') as map_file:
        _, index_to_img = pkl.load(map_file)

    features = np.zeros(shape=(len(index_to_img), 2048))

    model, layer = load_model()
    model.to(device)
    batch_size = 32

    for i in tqdm(range(0,len(index_to_img),batch_size)):
        end = min(len(index_to_img),i+batch_size)
        features[i:end] = get_image_features(index_to_img[i:end], model, layer)

    out = {}
    out['image_features'] = features
    
    ## save the created embeddings
    with open(params['embedding_output_path'], 'wb') as out_file:
        pkl.dump(out, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--mapping_file', default='image_index.pkl', help='This files contains the img_id to path mapping and vice versa')
    parser.add_argument('--embedding_output_path', default='../data/img_embedding.pkl', help='output pkl file with img features')

    args = parser.parse_args()
    params = vars(args)

    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))
    
    main(params)
