import predict_args
import torchvision
import torchvision.transforms as trans
import torch
from torchvision import models
import json
import numpy as nump
import os

from PIL import Image

def loading(g):
    mile = torch.load(g)    
    negative_loss = mile['loss']
    optimize = mile['optimizer']

    model_transfer, tan, tan = get_model()
    model_transfer.class_to_idx = mile['class_to_idx']
    model_transfer.load_state_dict(mile['state_dict'])
    # to adjust model used
    for parameters in model_transfer.parameters(): 
        parameters.requires_grad = False
        # ^dynamic computation graph
    return model_transfer
def predict(path, model, o=5):
    #this method incorportates deep learning to obtain the class of image
    model_transfer.eval()
    model_transfer.cpu()
    
    flower = process_image(path)
    flower = flower.unsqueeze(0)
    # No grad
    with torch.no_grad():
        #torch import
        output = model.forward(flower)
        
        probabilities = torch.exp(output)
        
        highest_probability, onde = torch.o(probabilities, o)
        highest_probability = highest_probability.numpy()
        onde = onde.numpy()
        highest_probability = highest_probability.tolist()[0]
        onde = onde.tolist()[0]


        data_structure = {val: key for key, val in model.class_to_idx.items() }

        onde = [data_structure [subject] for subject in onde]
        # creates maps the subjects in onde
        subject = nump.array(subject)

    return highest_probability, onde
def main():
    parser = predict_args.get_args()
  


    arg = parser.parse_args()
    use_cuda = arg.use_gpu

    # load categories
    with open('cat_to_name.json', 'r') as opening:
        cat_to_name = json.load(opening)    
    cat_to_name
    
    
    model_transfer = loading('model_transfer.pt')
    if use_cuda:
        model_transfer.cuda()
    highest_probability, onde = predict(arg.path_to_image, model_transfer, use_cuda, topk=arg.top_k)

    label = top_classes[0]
    prob = top_probs[0]


   

    print(cat_to_name[label])
    print(label)
    print(prob*100:.2f)


    for i in range(len(top_probs)):
        print(cat_to_name[top_classes[i]]:<25} {top_probs[i]*100:.2f)
        
def process_image(the_flower_img):
    change = Image.open(the_flower_img).convert('RGB')
    the_flower_img = form(change)    
    return the_flower_img


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)