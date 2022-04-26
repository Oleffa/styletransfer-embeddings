import torch
import urllib
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

batch_size = 10

transform = transforms.Compose([])
dataset_wikiart = ImageDataset('/media/oli/LinuxData/datasets/wikiart/image_face/', style='wikiart', transform=transform)
wikiart_loader = DataLoader(dataset_wikiart, batch_size=batch_size, shuffle=False)
dataset_celeba = ImageDataset('/media/oli/LinuxData/datasets/wikiart/img_align_celeba/', style='celeba', transform=transform)
celeba_loader = DataLoader(dataset_celeba, batch_size=batch_size, shuffle=False)
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


with torch.no_grad():
    xs = []
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    for i, d in enumerate(celeba_loader):

        # Early stopping
        if 'f5' in locals():
            if len(f5) * batch_size >= 1000:
                break

        x = d[0].to(device).float() / 255.0


        features5 = encoder(x, f'relu5_1')
        #rec5 = show(features5, 4)
        features4 = encoder(x, f'relu4_1')
        #rec4 = show(features4, 3)
        features3 = encoder(x, f'relu3_1')
        #rec3 = show(features3, 2)
        features2 = encoder(x, f'relu2_1')
        #rec2 = show(features2, 1)
        features1 = encoder(x, f'relu1_1')
        #rec1 = show(features1, 0)


        dtype = torch.float
        xs.append(x.cpu().to(dtype))
        #f1.append(features1.cpu().to(dtype))
        #f2.append(features2.cpu().to(dtype))
        #f3.append(features3.cpu().to(dtype))
        #f4.append(features4.cpu().to(dtype))
        f5.append(features5.cpu().to(dtype))

        print("{}/{}".format((i+1), len(wikiart_loader)))
torch.save(torch.cat(xs, dim=0), '../data/wikiart/celeba/celeba_input.pt')
#torch.save(torch.stack(f1, dim=0), '../data/wikiart/wikiart_embeddings_1.pt')
#torch.save(torch.stack(f2, dim=0), '../data/wikiart/wikiart_embeddings_2.pt')
#torch.save(torch.stack(f3, dim=0), '../data/wikiart/wikiart_embeddings_3.pt')
#torch.save(torch.stack(f4, dim=0), '../data/wikiart/wikiart_embeddings_4.pt')
torch.save(torch.cat(f5, dim=0), '../data/wikiart/celeba/celeba_embeddings_5.pt')

# TODO test load the embeddings and decode them to see if it worked

# Downstream experiment, train 5 VAES on the 5 sets of features (test individually and total comparison with
# all 5 vae networks in demo case)
