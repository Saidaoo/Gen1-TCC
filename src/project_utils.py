# Utils
import random, os, copy
import numpy as np
import torch
import itertools
from matplotlib import pyplot as plt
import albumentations as A

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
from os import system, name
from typing import Iterator, Tuple, Union

import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
# import segmentation_models_pytorch as smp

# palette = {
#     0 : (255, 0, 0),        # Desenvolvimento (vermelho)
#     1 : (38, 115, 0),       # Floresta Mata (verde escuro)
#     2 : (0, 255, 197),      # Piscina (ciano)
#     3 : (0, 0, 0),          # Sombra (preto)
#     4 : (133, 199, 126),    # Floresta Regeneração (verde claro)
#     5 : (255, 255, 0),      # Agricultura (amarelo)
#     6 : (255, 85, 0),       # Formação Rochosa (laranja)
#     7 : (115, 76, 0),       # Solo Exposto (marrom)
#     8 : (84, 117, 168),     # Água (azul escuro)
# }
palette = {
    0 : (255, 0, 0),        # Desenvolvimento (vermelho)
    1 : (38, 115, 0),       # Floresta Mata (verde escuro)
    # 2 : (0, 255, 197),      # Piscina (ciano)
    2 : (0, 0, 0),          # Sombra (preto)
    3 : (133, 199, 126),    # Floresta Regeneração (verde claro)
    4 : (255, 255, 0),      # Agricultura (amarelo)
    #5 : (255, 85, 0),       # Formação Rochosa (laranja)
    5 : (128, 128, 128),       # Formação Rochosa (laranja) novo
    #6 : (115, 76, 0),       # Solo Exposto (marrom)
    6 : (139, 69, 19),       # Solo Exposto (marrom) novo
    7 : (84, 117, 168),     # Água (azul escuro)
}

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        #print(np.array(c).shape)
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
    
def convert_from_color_torch(arr_3d: torch.Tensor, palette=invert_palette) -> torch.Tensor:
    assert arr_3d.ndim == 3 and arr_3d.shape[0] == 3, "Input deve ter shape [3, H, W]"
    _, H, W = arr_3d.shape
    arr_2d = torch.zeros((H, W), dtype=torch.uint8, device=arr_3d.device)

    for color, label in palette.items():
        # color é uma tupla (R, G, B), arr_3d é [3, H, W]
        r, g, b = color
        mask = (arr_3d[0] == r) & (arr_3d[1] == g) & (arr_3d[2] == b)
        arr_2d[mask] = label

    return arr_2d

def get_random_pos(img, window_shape):
    w, h = window_shape
    H, W = 2048, 2048             # agora H=altura, W=largura
    x1 = random.randint(0, W - w)     # permite x2 = W
    y1 = random.randint(0, H - h)
    x2 = x1 + w
    y2 = y1 + h
    return x1, x2, y1, y2

def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, reduction=reduction)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, reduction=reduction)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(img, stride=10, window_size=(20, 20)):
    """
    Gera coordenadas de janelas de tamanho window_size ao "deslizar" sobre a imagem img
    com passo (stride). Garante inclusão da última janela colada na borda, sem
    reatribuir a variável de iteração.
    """
    H, W = img.shape[:2]
    win_h, win_w = window_size

    # calcula todos os inícios em x, incluindo a borda final
    x_starts = list(range(0, max(1, H - win_h + 1), stride))
    if x_starts[-1] != max(0, H - win_h):
        x_starts.append(max(0, H - win_h))

    # calcula todos os inícios em y, incluindo a borda final
    y_starts = list(range(0, max(1, W - win_w + 1), stride))
    if y_starts[-1] != max(0, W - win_w):
        y_starts.append(max(0, W - win_w))

    # itera sem mutar as variáveis de loop
    for x0 in x_starts:
        for y0 in y_starts:
            yield x0, y0, win_h, win_w


def count_sliding_window(img_or_shape, stride=10, window_size=(20, 20)):

    # extrai altura/largura
    if hasattr(img_or_shape, "shape"):
        H, W = img_or_shape.shape[:2]
    else:
        H, W = img_or_shape[:2]

    win_h, win_w = window_size

    # número de passos em cada dimensão
    n_x = ( (H - win_h + stride - 1) // stride ) + 1
    n_y = ( (W - win_w + stride - 1) // stride ) + 1

    return n_x * n_y

def sliding_window_torch(
    img: Union[torch.Tensor, any],
    stride: int = 10,
    window_size: Tuple[int, int] = (20, 20)
) -> Iterator[Tuple[int, int, int, int]]:

    # extrai H, W da shape (caso esteja em [C, H, W] ou [H, W, C])
    shape = img.shape
    if len(shape) == 3:
        # detecta onde estão H e W
        if shape[0] in (1, 3):  # [C, H, W]
            H, W = shape[1], shape[2]
        else:                   # [H, W, C]
            H, W = shape[0], shape[1]
    elif len(shape) >= 2:
        H, W = shape[0], shape[1]
    else:
        raise ValueError("Entrada deve ter pelo menos 2 dimensões")

    win_h, win_w = window_size

    # calcula todos os inícios em x, incluindo a borda final
    x_starts = list(range(0, max(1, H - win_h + 1), stride))
    if x_starts[-1] != max(0, H - win_h):
        x_starts.append(max(0, H - win_h))

    # calcula todos os inícios em y, incluindo a borda final
    y_starts = list(range(0, max(1, W - win_w + 1), stride))
    if y_starts[-1] != max(0, W - win_w):
        y_starts.append(max(0, W - win_w))

    # itera sem mutar as variáveis de loop
    for x0 in x_starts:
        for y0 in y_starts:
            yield x0, y0, win_h, win_w


def count_sliding_window_torch(
    img_or_shape: Union[torch.Tensor, Tuple[int, int], list],
    stride: int = 10,
    window_size: Tuple[int, int] = (20, 20)
) -> int:

    # extrai H, W
    if isinstance(img_or_shape, torch.Tensor):
        shape = img_or_shape.shape
        if len(shape) == 3:
            # considera [C, H, W] ou [H, W, C]
            if shape[0] in (1, 3):
                H, W = shape[1], shape[2]
            else:
                H, W = shape[0], shape[1]
        else:
            H, W = shape[0], shape[1]
    else:
        # assume sequência [H, W]
        H, W = img_or_shape[0], img_or_shape[1]

    win_h, win_w = window_size

    # número de passos em cada dimensão
    n_x = ((H - win_h + stride - 1) // stride) + 1
    n_y = ((W - win_w + stride - 1) // stride) + 1

    return n_x * n_y

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
        
"""  Make optimizer routine"""
def make_optimizer(args, net):
    trainable = filter(lambda x: x.requires_grad, net.parameters()) # Only the parameters that requires gradient are passed to the optimizer

    #kwargs['lr'] = args['lr']
    #kwargs['weight_decay'] = args['weight_decay']

    if args['optimizer'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': 0.9,
            'nesterov': True
        }
    elif args['optimizer'] == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon'],
            'weight_decay': args['weight_decay']
        }
    elif args['optimizer'] == 'ADAMW':
        optimizer_function = optim.AdamW
        kwargs = {
            'lr': args['lr'],
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon'],
            'weight_decay': args['weight_decay']
        }
    elif args['optimizer'] == 'RADAM':
        optimizer_function = optim.RAdam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon'],
            'weight_decay': args['weight_decay'],
            'decoupled_weight_decay': True
        }
    elif args['optimizer'] == 'NADAM':
        optimizer_function = torch.optim.NAdam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon'],
            'momentum_decay': 4e-3,
            'weight_decay': args['weight_decay'],
            'decoupled_weight_decay': False
        }
    elif args['optimizer'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args['epsilon']}
    
    return optimizer_function(trainable, **kwargs)



""" Make scheduler routine """
def make_scheduler(args, optimizer):
    if args['type'] == 'multi':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=args['gamma']
        )
    elif args['type'] == 'CLR':
        scheduler = lrs.CyclicLR(optimizer, base_lr = 1e-5, max_lr = 1e-2)
    elif args['type'] == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',             
            factor=0.5,              
            patience=3,             
            #threshold=1e-4,          
            #threshold_mode='rel',   
            cooldown=3,              # evita reduções seguidas
            min_lr=1e-6,             # mínimo aceitável
            verbose=True
        )
    else:
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    return scheduler


def calculate_cm(predictions, labels, label_values = None, normalize = None):
    return confusion_matrix(labels, predictions, labels=label_values, normalize=normalize)


""" Global acurracy metric calculation """
def global_accuracy(predictions, labels):
    # Calculate confusion matrix
    cm = calculate_cm(predictions, labels)
    # Sum all values in main diagonal
    main_diagonal = sum([cm[i][i] for i in range(len(cm))])
    # return TP+TN / TP+TN+FN x 100%
    return 100 * float(main_diagonal) / np.sum(cm)

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    # dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image.cpu().numpy().transpose(1,2,0))
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def new_acc(predictions, gts, label_values, all=False, filepath=None):
    #print(f'gts.shape={np.array(gts).shape}, predictions.shape={np.array(predictions).shape}')
    
    cm = confusion_matrix(
            gts,
            predictions,
            labels=range(len(label_values)))
    
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    
    return accuracy

def metrics(predictions, gts, label_values, all=False, filepath=None):
    
    txt = []
    cm = confusion_matrix(
            gts,
            predictions,
            labels=range(len(label_values)))
    
    # plt.show()
    # plot_confusion_matrix(
    #     cm           = cm, 
    #     normalize    = False,
    #     target_names = range(len(label_values)),
    #     title        = "Confusion Matrix"
    # )
    fig, ax = plot_confusion_matrix(conf_mat=cm * 100,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=label_values)
    fig.savefig(f"./{filepath}/cm_all" if all else f'./{filepath}/cm' , dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)
    
    print("Confusion matrix :")
    print(cm)
    txt.append("Confusion Matrix")
    txt.append(cm)
    
    print("---")
    txt.append("------")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    txt.append("{} pixels processed".format(total))
    #txt.append("Total accuracy : {}%".format(accuracy))
    
    print("---")
    txt.append("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    txt.append('\nF1Score :')
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))
        txt.append("{}: {}".format(label_values[l_id], score))
    print(f"mean F1Score : {100.0*np.mean(F1Score)}")
    txt.append(f"mean F1Score : {100.0*np.mean(F1Score)}")
    print("Total accuracy : {}%".format(accuracy))
    txt.append("Total accuracy : {}%".format(accuracy))

    #print("---")
    #txt.append("---")
        
    # Compute kappa coefficient
    #total = np.sum(cm)
    #pa = np.trace(cm) / float(total)
    #pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    #kappa = (pa - pe) / (1 - pe);
    #print("Kappa: " + str(kappa))
    #txt.append("\nKappa: " + str(kappa))
    
    with open(f"{filepath}/metrics_test.txt", "w") as f:
        for line in txt:
            f.write(str(line) + "\n")
    
    return accuracy

# define our clear function
def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
        
def plot_confusion_matrix_local(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save:
        fig.savefig(f"./tmp/cm", dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean, std

def save_test(acc, all_preds, all_gts, path=None):
    try:
        print('************** save test results **************')
        if path is None:
            path = './segnet256_test_result.npz'
            
        np.savez_compressed(path, {
            'acc': acc,
            'all_preds': all_preds,
            'all_gts': all_gts,
        })
    except:
        print('[AVISO] Erro ao salvar os resultados do teste!')
        pass
    
def load_test(path=None):
    assert os.path.exists(path), "{} cant be opened".format(path)
    
    data = np.load(path, allow_pickle=True)
    return data.item()

def save_loss_weights(data, path=None):
    if path is None:
        path = './loss_weights.npy'
    np.save(path, data)
        
def load_loss_weights(path=None):
    if path is None:
        path = './loss_weights.npy'
    x = np.load(path, allow_pickle=True)
    return x.item()

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# import torchvision.models as models
# import torch.nn as nn
# def build_model_efficientnet_b0(num_classes, pretrained=True, fine_tune=True):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights')
#     else:
#         print('[INFO]: Not loading pre-trained weights')
#     model = models.efficientnet_b0(pretrained=pretrained)
#     if fine_tune:
#         print('[INFO]: Fine-tuning all layers...')
#         for params in model.parameters():
#             params.requires_grad = True
#     elif not fine_tune:
#         print('[INFO]: Freezing hidden layers...')
#         for params in model.parameters():
#             params.requires_grad = False
#     # Change the final classification head.
#     model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
#     return model