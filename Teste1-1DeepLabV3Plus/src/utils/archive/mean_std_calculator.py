import sys
sys.path.append('D:\\Projetos\\icmbio\\')

import os
from PIL import Image
from skimage import io
import numpy as np
import pandas as pd
from utils import convert_from_color, convert_to_color, save_loss_weights

# EXT = 'tif'
DEV = 5

class MeanStdCalculator():
    
    def __init__(self, train_images: str, dev=False, filename = 'mean_std'):
        self.train_images = train_images
        self.filename = f'{filename}_dev' if dev else filename
        
        # print(f'--- Carregando as {len(train_images)} imagens ---')
        # if dev:
        #     self.images = [io.imread(file) for file in train_images[0:DEV]]
        # else:
        #     self.images = [io.imread(file) for file in train_images]
        if dev:
            self.images = train_images[0:DEV]
        else:
            self.images = train_images

    def compute_mean_and_std(self):
        from tqdm import tqdm
        mean_r = 0
        mean_g = 0
        mean_b = 0

        for img in tqdm(self.images):
            img = np.asarray(Image.open(img)) # change PIL Image to numpy array
            mean_b += np.mean(img[:, :, 0])
            mean_g += np.mean(img[:, :, 1])
            mean_r += np.mean(img[:, :, 2])

        mean_b /= len(self.images)
        mean_g /= len(self.images)
        mean_r /= len(self.images)

        diff_r = 0
        diff_g = 0
        diff_b = 0

        N = 0

        for img in tqdm(self.images):
            img = np.asarray(Image.open(img)) # change PIL Image to numpy array

            diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
            diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
            diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

            N += np.prod(img[:, :, 0].shape)

        std_b = np.sqrt(diff_b / N)
        std_g = np.sqrt(diff_g / N)
        std_r = np.sqrt(diff_r / N)

        mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
        std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
        
        return mean, std
  
# Função para calcular Mean e STD das imagens em um diretório
if __name__=='__main__':

    label_dir = os.path.join('D:\\datasets\\ICMBIO_NOVO\\all', 'images')
    train_images = pd.read_table('train_images.txt',header=None).values
    train_images = [os.path.join(label_dir, f[0]) for f in train_images]

    ms = MeanStdCalculator(dev=False,train_images=train_images)
    mean, std = ms.compute_mean_and_std()
    print(f'Mean -> {mean}, Std -> {std}')
    