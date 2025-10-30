import sys
sys.path.append('D:\\Projetos\\icmbio\\')

import os
# import glob
from skimage import io
import numpy as np
import pandas as pd
from project_utils import convert_from_color, convert_to_color, save_loss_weights

# EXT = 'tif'
DEV = 10

class WeightsCalculator():
    
    def __init__(self, train_labels: str, classes: list, dev=False, filename = 'loss_weights'):
        self.train_labels = train_labels
        self.classes = classes
        self.filename = f'{filename}_dev' if dev else filename
        self.C = len(classes) # quantidade de classes
        self.N = None # Pixels por classe
        self.M = None # Soma dos pixels de todas as imagens
        
        # assert os.path.exists(label_dir), "{}. Diretório não existe".format(label_dir)
        # image_list = glob.glob(f"{label_dir}/*.{EXT}")

        print(f'--- Carregando as {len(train_labels)} imagens ---')
        if dev:
            self.images = [io.imread(file) for file in train_labels[0:DEV]]
        else:
            self.images = [io.imread(file)[:,:,:3] for file in train_labels]
        
        self.load()
        
    '''Calcular os pixels por classe de todas as imagens e os pixels totais'''
    def load(self) -> None:
        print(f'--- Convertendo imagem para Paleta de Cores ---')
        image_color = [convert_from_color(image) for image in self.images]  

        pixels_by_image_class = []
        for idx, image in enumerate(image_color):
            print(f'--- Somando pixels da imagem {idx+1}/{len(image_color)} ---')
            gen = ([image.ravel() == i for i in np.arange(self.C)])
            pixels_by_image_class.append(np.sum(list(gen), axis=1))
            
        self.N = np.sum(pixels_by_image_class, axis=0, dtype=np.int64)
        self.M = np.sum(pixels_by_image_class)
        df = pd.DataFrame(data=pixels_by_image_class)
        df.to_csv('pesos.csv', encoding='utf-8', sep=';', decimal=',')
        
    def calculate_and_save(self, normalize=True) -> list:
        prod = np.array([self.C*n for n in self.N], dtype=np.int64)
        # weights = np.divide(self.M, prod, where=prod!=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.true_divide(self.M, prod)
            weights[weights == np.inf] = 0
            weights = np.nan_to_num(weights)
        if normalize:
            weights_norm = weights / weights.sum()
            # weights_norm = np.divide(weights, np.linalg.norm(weights))
            # weights_norm = self.__normalize(weights)

        # Format decimals
        weights, weights_norm = self.__formatDecimal(weights), self.__formatDecimal(weights_norm)
        # save to file and csv
        self.__save({'weights': weights, 'weights_norm': weights_norm})
        return weights, weights_norm
    
    def __save(self, data) -> None:
        
        df = pd.DataFrame(data=data, index=self.classes)
        df.to_csv(f'{self.filename}.csv', encoding='utf-8', sep=';', decimal=',')
        save_loss_weights(data, f'{self.filename}.npy')
    
    # explicit function to normalize array
    # def __normalize(self, arr, t_min=0, t_max=1):
    #     norm_arr = []
    #     diff = t_max - t_min
    #     diff_arr = max(arr) - min(arr)
    #     for i in arr:
    #         temp = (((i - min(arr))*diff)/diff_arr) + t_min
    #         norm_arr.append(temp)
    #     return norm_arr
    
    def __formatDecimal(self, arr, decimals=4):
        return np.around(arr, decimals=decimals)
  
# Função para calcular o peso de cada classe em um conjunto de imagens (labels)        
# if __name__=='__main__':

#     label_dir = os.path.join('D:\\datasets\\ICMBIO_NOVO\\all', 'label')
#     train_labels = pd.read_table('train_labels.txt',header=None).values
#     train_labels = [os.path.join(label_dir, f[0]) for f in train_labels]
    
#     wc= WeightsCalculator(
#         train_labels=train_labels,
#         classes=["Urbano", "Mata", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"],
#         dev=False,
#         filename='loss_weights'
#     )
#     weights, weights_norm = wc.calculate_and_save()
    
#     print(weights)
#     print(weights_norm)
    
#     # N = [114042226, 497764795, 1795239, 56643355, 173535660, 21323789, 29305581, 0]
#     # C = 9
#     # M = 910163968
    
#     # pesos = np.divide(M, [C*n for n in N])
#     # print(pesos)
#     # print(np.divide(pesos, np.linalg.norm(pesos)))
    
#     # classes = ["Urbano", "Mata", "Piscina", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"]
#     # weights = [0.88677092, 4.92204386, 56.33195902, 1.7853697, 0.58275821, 4.74255911, 3.45085565, 13.91675129, 11.91636944]
#     # weights_norm = [0.01483752, 0.08235605, 0.942551, 0.02987295, 0.00975076, 0.07935289, 0.05774, 0.23285624, 0.19938568]
    
#     # df = pd.DataFrame({'weights': weights, 'weights_norm': weights_norm}, columns=classes)
#     # print(df)