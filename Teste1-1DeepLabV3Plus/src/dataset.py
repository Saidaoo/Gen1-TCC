import time
import torch
import os
import cv2
import random
import numpy as np
from skimage import io, img_as_float64
from project_utils import convert_from_color, get_random_pos, convert_to_color
import matplotlib.pyplot as plt
import albumentations as A

EXTS = ['tif', 'tiff', 'jpg', 'png']

def cutmix(img1, lbl1, img2, lbl2, alpha=1.0):
    """
    Aplica CutMix em (img1, lbl1) com (img2, lbl2).
    img*: HxWxC (numpy), lbl*: HxW
    """
    
    #ridx = random.randint(0, len(self.data_files) - 1)
    #img2 = io.imread(self.data_files[ridx])[...,:3]
    #lbl2 = convert_from_color(io.imread(self.label_files[ridx])[...,:3]).astype('uint8')
    #x1, x2, y1, y2 = get_random_pos(data, self.window_size)

    #img2  = img2[x1:x2, y1:y2,:]
    #lbl2 = lbl2[x1:x2, y1:y2]
    #img2, lbl2 = cv2.resize(img2, (img.shape[1], img.shape[0])), cv2.resize(lbl2, (lbl.shape[1], lbl.shape[0]))
    
    lam = np.random.beta(alpha, alpha)
    H, W, _ = img1.shape

    # Define o ret�ngulo de recorte
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # aplica o recorte na imagem e no r�tulo
    img1[y1:y2, x1:x2, :] = img2[y1:y2, x1:x2, :]
    lbl1[y1:y2, x1:x2] = lbl2[y1:y2, x1:x2]

    return img1, lbl1

# Dataset class
class DatasetIcmbio(torch.utils.data.Dataset):

    def __init__(
        self,
        data_files,
        label_files,
        edge_files=None,          # <-- arquivos de bordas (opcional)
        window_size=224,
        n_channels=3,
        cache=True,
        augmentation=True
    ):
        super().__init__()
        self.data_files  = data_files
        self.label_files = label_files
        self.edge_files  = edge_files
        self.window_size = window_size
        self.n_channels  = n_channels
        self.cache       = cache
        self.augmentation = augmentation

        # sanity check
        all_files = self.data_files + self.label_files
        if self.edge_files is not None:
            all_files += self.edge_files
        for f in all_files:
            if not os.path.isfile(f):
                raise KeyError(f"'{f}' nao encontrado!")

        # caches
        self.data_cache_  = {}
        self.label_cache_ = {}
        self.edge_cache_  = {}  # cache para bordas

    def __len__(self):
        return 21*100*5

    def get_dataset(self):
        return self.data_files, self.label_files, self.edge_files

    #@classmethod
    def data_augmentation(self, img, lbl, img2=None, lbl2=None, apply_prob=0.5, test_mode=False):
        #if not arrays or len(arrays) < 2:
        #    raise ValueError("Esperado ao menos imagem e label.")

        #img, lbl = arrays[0], arrays[1]
        #img = img.transpose(1, 2, 0)  # [3,h,w] ? [h,w,3]

        if not test_mode:  # modo normal (pipeline)
            if random.random() >= apply_prob:
                return img.transpose(2,0,1).astype('float32') / 255.0, lbl
            '''
            aug = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.RandomRotate90(p=1),
                    A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=1),
                    A.Downscale(scale_min=0.5, scale_max=0.5, interpolation=cv2.INTER_LINEAR, p=1)
                ], p=0.8),
                A.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.2, hue=0.1, p=0.7),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
            ])
            '''
            aug = A.RandomRotate90(p=1)
            augmented = aug(image=img, mask=lbl)
            img_aug = augmented["image"].transpose(2,0,1).astype('float32') / 255.0
            lbl_aug = augmented["mask"]
            
            #img_aug, lbl_aug = cutmix(img.copy(), lbl.copy(), img2, lbl2)
            #img_aug = img_aug.transpose(2,0,1).astype('float32') / 255.0

            #print(img_aug.dtype, lbl_aug.dtype)
            return img_aug, lbl_aug

        else:  # modo de teste ? aplica cada transforma��o separada e plota
            transforms = {
                "Flip Horizontal": A.HorizontalFlip(p=1),
                "Flip Vertical": A.VerticalFlip(p=1),
                "Rotate90": A.RandomRotate90(p=1),
                "Rotate [-30,30]": A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=1),
                "Downscale 2x": A.Downscale(scale_min=0.5, scale_max=0.5, interpolation=cv2.INTER_LINEAR, p=1),
                "ColorJitter": A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1, p=1),
                "Gaussian Blur": A.GaussianBlur(blur_limit=(3, 5), p=1),
                "Gaussian Noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1)
            }
            
            for name, t in transforms.items():
                augmented = t(image=img, mask=lbl)
                img_t, lbl_t = augmented["image"], augmented["mask"]

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img_t)
                axs[0].set_title(f"{name} - Imagem")
                axs[0].axis("off")
                axs[1].imshow(lbl_t, cmap="tab20")
                axs[1].set_title(f"{name} - Label")
                axs[1].axis("off")
                #plt.show()
                fig.savefig(f"/prj/posgrad/pasc/ICMBIO/dataset_35/augTest/{name}.png", dpi=150)

            img_t, lbl_t = cutmix(img.copy(), lbl.copy(), img2.copy(), lbl2.copy())
                
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img_t)
            axs[0].set_title(f"CutMix - Imagem")
            axs[0].axis("off")
            axs[1].imshow(lbl_t, cmap="tab20")
            axs[1].set_title(f"CutMix - Label")
            axs[1].axis("off")
            #plt.show()
            fig.savefig(f"/prj/posgrad/pasc/ICMBIO/dataset_35/augTest/cutmix.png", dpi=150)

            return img.transpose(2,0,1).astype('float32')/255.0, lbl

    def __getitem__(self, idx):
        ridx = random.randint(0, len(self.data_files) - 1)

        # carrega/imprime data
        if ridx in self.data_cache_:
            data = self.data_cache_[ridx]
        else:
            img = io.imread(self.data_files[ridx])[...,:3]
            data = img.astype('uint8')#(img.transpose(2,0,1) / 255.0).astype('float32')
            if self.cache:
                self.data_cache_[ridx] = data

        # carrega label
        if ridx in self.label_cache_:
            label = self.label_cache_[ridx]
        else:
            lbl_img = io.imread(self.label_files[ridx])[...,:3]
            label = convert_from_color(lbl_img).astype('uint8')
            if self.cache:
                self.label_cache_[ridx] = label

        # carrega bordas se fornecido
        if self.edge_files is not None:
            if ridx in self.edge_cache_:
                edges = self.edge_cache_[ridx]
            else:
                edge_img = io.imread(self.edge_files[ridx])
                # se for RGB (3 canais), extraia o primeiro; se for grayscale (2D), use direto
                if edge_img.ndim == 3:
                    edges = (edge_img[..., 0] / 255.0).astype('float32')
                elif edge_img.ndim == 2:
                    edges = (edge_img / 255.0).astype('float32')
                else:
                    raise ValueError(f"Unexpected edge_img.ndim = {edge_img.ndim}")
                if self.cache:
                    self.edge_cache_[ridx] = edges
        else:
            edges = None

        # extrai patch aleat�rio
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)

        data_p  = data[x1:x2, y1:y2,:]
        label_p = label[x1:x2, y1:y2]
        if edges is not None:
            edge_p = edges[x1:x2, y1:y2]


        #img2 = io.imread(self.data_files[ridx])[...,:3]
        #lbl2 = convert_from_color(io.imread(self.label_files[ridx])[...,:3]).astype('uint8')
        #print("1", data_p.shape, label_p.shape, data_p.dtype, label_p.dtype)
        # data augmentation
        if self.augmentation:
            if edges is not None:
                data_p, label_p, edge_p = self.data_augmentation(
                    data_p, label_p, edge_p, img2, lbl2, edg2
                )
            else:
                '''
                ridx = random.randint(0, len(self.data_files) - 1)
                if ridx in self.data_cache_:
                    data = self.data_cache_[ridx]
                else:
                    img = io.imread(self.data_files[ridx])[...,:3]
                    data = img.astype('uint8')#(img.transpose(2,0,1) / 255.0).astype('float32')
                    if self.cache:
                        self.data_cache_[ridx] = data
                if ridx in self.label_cache_:
                    label = self.label_cache_[ridx]
                else:
                    lbl_img = io.imread(self.label_files[ridx])[...,:3]
                    label = convert_from_color(lbl_img).astype('uint8')
                    if self.cache:
                        self.label_cache_[ridx] = label
                x1, x2, y1, y2 = get_random_pos(data, self.window_size)
                img2  = data[x1:x2, y1:y2,:]
                lbl2 = label[x1:x2, y1:y2]
                #print("2", img2.shape, lbl2.shape, img2.dtype, lbl2.dtype)
                '''
                data_p, label_p = self.data_augmentation(
                    data_p, label_p#, img2, lbl2
                )
        else:
            data_p = (data_p.transpose(2,0,1) / 255.0).astype('float32')
        #print("3", data_p.shape, label_p.shape, data_p.dtype, label_p.dtype)

        # convers�o para tensores
        data_t = torch.from_numpy(data_p)
        label_t = torch.from_numpy(label_p)
        if edges is not None:
            edge_t = torch.from_numpy(edge_p)
            return data_t, label_t, edge_t
        else:
            return data_t, label_t