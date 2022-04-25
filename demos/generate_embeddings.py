import torch
import pickle
from model_style_transfer import MultiLevelAE
import matplotlib.pyplot as plt
from torchvision import transforms 
from PIL import Image
import os
import torchvision.transforms as transforms

from dataloader import ImageDataset
from torch.utils.data import DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))
models_path = dir_path + '/../data/models'
url_models = 'https://remi.flamary.com/download/models/'
trans = transforms.Compose([transforms.ToTensor()])

lst_model_files=[ "decoder_relu1_1.pth",
                  "decoder_relu2_1.pth",
                  "decoder_relu3_1.pth",
                  "decoder_relu4_1.pth",
                  "decoder_relu5_1.pth",
                  "vgg_normalised_conv5_1.pth"]
trans = transforms.Compose([transforms.ToTensor()])

# test if models already downloaded
for m in lst_model_files:
    if not os.path.exists(models_path+'/'+m):
        print('Downloading model file : {}'.format(m))
        urllib.request.urlretrieve(url_models+m,models_path+'/'+m)


if torch.cuda.is_available():
    device = torch.device(f'cuda')
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'

show_rec = False
model = MultiLevelAE(models_path)
model = model.to(device)
encoder = model.encoder
decoders = [model.decoder1, model.decoder2, model.decoder3, model.decoder4, model.decoder5]
print("Model loaded")

#transform = transforms.Compose([transforms.Resize((320, 192))])
transform = transforms.Compose([])
dataset_realism = ImageDataset('../data/wikiart/realism/', style='realism', transform=transform)
realism_loader = DataLoader(dataset_realism, batch_size=1, shuffle=False)
dataset_impressionism = ImageDataset('../data/wikiart/impressionism/', style='impressionism', transform=transform)
impressionism_loader = DataLoader(dataset_impressionism, batch_size=1, shuffle=False)
print("Datasets loaded")

class Datapoint():
    def __init__(self, input_image, embeddings, reconstructions):
        self.input_image = input_image
        self.embeddings = embeddings
        #self.reconstructions = [(i * 255.0).to(torch.uint8) for i in reconstructions]

def show(embedding, dec):
    dec = decoders[dec](embedding)
    if show_rec:
        plt.imshow(dec[0].permute(1,2,0))
        plt.show()
    return dec[0]

datapoints = []
with torch.no_grad():
    for i, d in enumerate(realism_loader):
        x = d[0].float() / 255.0
        features5 = encoder(x, f'relu5_1')
        rec5 = show(features5, 4)
        features4 = encoder(x, f'relu4_1')
        rec4 = show(features4, 3)
        features3 = encoder(x, f'relu3_1')
        rec3 = show(features3, 2)
        features2 = encoder(x, f'relu2_1')
        rec2 = show(features2, 1)
        features1 = encoder(x, f'relu1_1')
        rec1 = show(features1, 0)
        print(features5.numpy().nbytes / 1000000.0)
        print(features4.numpy().nbytes / 1000000.0)
        print(features3.numpy().nbytes / 1000000.0)
        print(features2.numpy().nbytes / 1000000.0)
        print(features1.numpy().nbytes / 1000000.0)
        d = Datapoint(d[0][0], [features1, features2, features3, features4, features5], \
                [rec1, rec2, rec3, rec4, rec5])
        datapoints.append(d)
        print("{}/{}".format(i+1, len(realism_loader)))
        with open('../data/wikiart/realism.pkl', 'wb') as f:
            pickle.dump(datapoints, f)
