import pickle as pkl
import torch
import torch.nn as nn

def filterOutput(outputs, labels, label_embeds, PAD_token_ind):
    mask = (labels != PAD_token_ind).squeeze()
    labels, label_embeds, outputs = labels[mask], label_embeds[mask], outputs[mask]

    return labels, label_embeds, outputs

def maskedLoss(label_embeds, outputs, criterion):
    num_tokens = label_embeds.shape[0]

    loss = criterion(label_embeds, outputs)
    loss = torch.sum(loss)/num_tokens
    
    return loss

# def accuracy(output_embeds, label_embeds, criterion):
    

def word_accuracy(output_embeds, txt_embeds, labels):
    """
        Input:
            output_embeds - Flattened Output of the decoder with padding token removed
            txt_embeds - Text embeds for the entire vocab
            labels - Flattened labels
        
        Output:
            Word Level Accuracy

    TODO(Jay): Address this issue
    There is a problem, since the embedding of SOS == EOS
    So for EOS it will predict index 2 which is index for SOS coz it comes first

    """
    num_words = output_embeds.shape[0]

    similarity = output_embeds.mm(txt_embeds.t())
    output_indices = similarity.argmax(1)
    
    labels = labels.squeeze()
    output_indices = output_indices.squeeze()
    assert output_indices.shape == labels.shape

    correct = torch.sum(output_indices == labels).float()
    acc = 100*(correct/num_words)
    return acc

# def accuracy(output_embeds, txt_embeds, labels, PAD_token_ind):
#     batch_size = output_embeds.shape[0]

#     for i in range(batch_size):
#         output_embeds

#     similarity = output_embeds.mm(txt_embeds.t())
#     output_indices = similarity.argmax(1)


if __name__ == "__main__":
    labels = torch.tensor([2, 45, 65, 71, 32, 3])
    correct_outputs = [2, 45, 65, 71, 32, 3]
    wrong_outputs = [3, 32, 71, 65, 45, 2]

    with open('./data/txt_embed.pkl', 'rb') as f:
        txt_embeds = torch.from_numpy(pkl.load(f))
    
    correct_acc = word_accuracy(txt_embeds[correct_outputs], txt_embeds, labels)
    wrong_acc = word_accuracy(txt_embeds[wrong_outputs], txt_embeds, labels)

    print('Accuracy for correct outputs: {}'.format(correct_acc))
    print('Accuracy for wrong outputs: {}'.format(wrong_acc))
