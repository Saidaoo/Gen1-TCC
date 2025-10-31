import warnings
from pathlib import Path
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
from project_utils import clear, count_sliding_window, count_sliding_window_torch, make_optimizer, make_scheduler, CrossEntropy2d, accuracy, metrics, new_acc, save_test, sliding_window, sliding_window_torch, grouper, convert_from_color, convert_to_color, convert_from_color_torch, global_accuracy
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, TverskyLoss, JaccardLoss
from jaccard_ce_loss import JaccardCELoss
import torch.nn.functional as F
import torch.nn as nn
import kornia
import kornia.filters as KF  
from torchvision import io as ioG
from sklearn.metrics import confusion_matrix
import math
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, JaccardIndex
from typing import Optional, Sequence
# --- Utilitário para ativar dropout em inferência (MC Dropout) sem ativar BatchNorm updates ---
def enable_mc_dropout(model: nn.Module):
    """Mantém model em eval(), mas ativa apenas camadas Dropout para MC samples.
       Uso típico:
         model.eval()
         enable_mc_dropout(model)
         # agora chamamos forward N vezes para estimar incerteza
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        # mantém BatchNorm em eval() explicitamente — já está em eval() pelo model.eval()


class F1Accumulator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # matriz de confusão global acumulada
        self.global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, gts: np.ndarray) -> None:
        # Achata ambos
        p = preds.ravel()
        t = gts.ravel()
        # Matriz de confusão desta chamada
        cm = confusion_matrix(t, p, labels=list(range(self.num_classes)))
        # Acumula
        self.global_cm += cm

    def compute(self) -> (np.ndarray, float):
        cm = self.global_cm
        f1_per_class = np.zeros(self.num_classes, dtype=float)
        for i in range(self.num_classes):
            denom = np.sum(cm[i, :]) + np.sum(cm[:, i])
            if denom > 0:
                f1_per_class[i] = 2. * cm[i, i] / denom
            else:
                # classe nunca vista => F1 = 0
                f1_per_class[i] = 0.0
        macro_f1 = np.mean(f1_per_class)
        return f1_per_class, macro_f1

def multiclass_f1_score_numpy(preds: np.ndarray,
                              targets: np.ndarray,
                              num_classes: int,
                              ignore_index: int = None) -> float:

    # Achata tudo em vetor
    preds_flat = preds.ravel()
    targets_flat = targets.ravel()

    # Se for para ignorar algum índice, filtra esses elementos
    if ignore_index is not None:
        mask = targets_flat != ignore_index
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]

    f1_scores = []
    for cls in range(num_classes):
        # Verdadeiros positivos: previu cls e alvo era cls
        tp = np.logical_and(preds_flat == cls, targets_flat == cls).sum()
        # Falsos positivos: previu cls mas alvo não era cls
        fp = np.logical_and(preds_flat == cls, targets_flat != cls).sum()
        # Falsos negativos: previu diferente de cls mas alvo era cls
        fn = np.logical_and(preds_flat != cls, targets_flat == cls).sum()

        denom = (2 * tp + fp + fn)
        if denom == 0:
            # Nenhum exemplo desta classe em preds nem targets: define F1 como 0
            f1 = 0.0
        else:
            f1 = 2 * tp / denom
        if (f1):
            f1_scores.append(f1)

    # Média não ponderada das classes
    return float(np.mean(f1_scores))

def l_inv_tv_borda(
    outputs: torch.Tensor,
    edges: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Inverse Total Variation Loss only on border pixels, retornando valores positivos.
    """
    probs = F.softmax(outputs, dim=1)
    B, C, H, W = probs.shape
    mask = edges.unsqueeze(1).float()

    diff_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    diff_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])

    mask_h = mask[:, :, 1:, :]
    mask_w = mask[:, :, :, 1:]
    masked_h = diff_h * mask_h
    masked_w = diff_w * mask_w

    loss_h = masked_h.sum(dim=1)
    loss_w = masked_w.sum(dim=1)

    if reduction == "none":
        pad_h = F.pad(loss_h, (0, 0, 0, 1))  # adiciona uma linha zero na base
        pad_w = F.pad(loss_w, (0, 1, 0, 0))  # adiciona uma coluna zero à direita
        inv_tv_map = pad_h + pad_w          # **sem** o sinal negativo
        return inv_tv_map                   # [B, H, W]

    # soma global **sem** o sinal negativo
    inv_tv = loss_h.sum() + loss_w.sum()

    if reduction == "sum":
        return inv_tv

    elif reduction == "mean":
        count = (mask_h.sum() + mask_w.sum()) * C
        return inv_tv / (count + 1e-6)

    else:
        raise ValueError(f"Reduction '{reduction}' not supported.")

def tv_loss_in_calm_zone(outputs: torch.Tensor,
                         edges: torch.Tensor,
                         reduction: str = 'mean') -> torch.Tensor:
    """
    Total Variation Loss only on the 'calm zone' (onde edges==0).

    Args:
        outputs (torch.Tensor): logits da UNet com shape [B, C, H, W].
        edges   (torch.Tensor): mapa de bordas com shape [B, H, W], valores 0/1.
        reduction (str): 'mean', 'sum' ou 'none'.

    Returns:
        torch.Tensor:
            - se reduction='none': tensor de shape [B] com o TV loss de cada amostra;
            - se reduction='sum': escalar somatório de todas as amostras;
            - se reduction='mean': escalar média (normalizada por pixel ativo) sobre todas as amostras.
    """
    B, C, H, W = outputs.shape

    # 1) converte logits em probabilidades
    probs = F.softmax(outputs, dim=1)  # [B, C, H, W]

    # 2) máscara da zona calma: 1 onde edges==0, 0 onde edges==1
    mask = (1.0 - edges.float()).unsqueeze(1)  # [B, 1, H, W]

    # 3) diferenças horizontais e verticais
    diff_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])  # [B, C, H-1, W]
    diff_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])  # [B, C, H, W-1]

    # 4) aplica máscara (ajustada para cada dimensão)
    mask_h = mask[:, :, 1:, :]  # [B, 1, H-1, W]
    mask_w = mask[:, :, :, 1:]  # [B, 1, H, W-1]

    # Soma por canal e espaços para cada amostra
    # tv_h and tv_w ficam com shape [B]
    tv_h = (diff_h * mask_h).sum(dim=[1,2,3])  # [B]
    tv_w = (diff_w * mask_w).sum(dim=[1,2,3])  # [B]
    tv_per_sample = tv_h + tv_w                 # [B]

    if reduction == 'none':
        return tv_per_sample

    elif reduction == 'sum':
        # soma todas as amostras
        return tv_per_sample.sum()

    elif reduction == 'mean':
        # normaliza por pixel ativo em cada amostra, depois tira média sobre batch
        # contar pixels ativos por amostra e por canal
        count_h = mask_h.sum(dim=[1,2,3]) * C    # [B]
        count_w = mask_w.sum(dim=[1,2,3]) * C    # [B]
        count_per_sample = count_h + count_w     # [B]
        # evita divisão por zero
        loss_norm = tv_per_sample / (count_per_sample + 1e-6)  # [B]
        return loss_norm.mean()

    else:
        raise ValueError(f"Reduction '{reduction}' not supported. Use 'none', 'sum' or 'mean'.")

def normalized_pixel_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calcula a entropia normalizada de cada pixel a partir de logits.

    Args:
        logits (torch.Tensor): Tensor de formato [B, C, H, W] contendo
                               scores não normalizados (logits).

    Retorna:
        torch.Tensor: Tensor [B, H, W] com entropia normalizada em [0, 1].
    """
    # logits: [B, C, H, W]
    # 1) Probabilidades via softmax ao longo de C
    probs = F.softmax(logits, dim=1)  # :contentReference[oaicite:3]{index=3}

    # 2) Log-probabilidades (estável) via log_softmax
    log_probs = F.log_softmax(logits, dim=1)  # :contentReference[oaicite:4]{index=4}

    # 3) Entropia por pixel: H = -sum_c p * log p
    entropy = - (probs * log_probs).sum(dim=1)  # resultado [B, H, W] :contentReference[oaicite:5]{index=5}

    # 4) Normalização por log(C)
    C = logits.shape[1]
    # Criamos um tensor escalar de log(C) no mesmo dispositivo e tipo de `entropy`
    log_C = torch.log(torch.tensor(C, device=logits.device, dtype=entropy.dtype))
    normalized_entropy = entropy / log_C

    return normalized_entropy

def uncertainty_in_calm_zone(
    logits: torch.Tensor,
    edges: torch.Tensor,
    mode: str = "entropy",
    eps: float = 1e-12
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)

    B, C, H, W = logits.shape

    if mode == "max_confidence":
        p_max, _ = probs.max(dim=1)          
        uncertainty = (1.0 - p_max) / (1.0 - 1.0 / C)

    elif mode == "entropy":
        ent = - (probs * torch.log(probs + eps)).sum(dim=1) 
        H_max = torch.log(torch.tensor(C, device=logits.device, dtype=ent.dtype))
        uncertainty = ent / H_max

    else:
        raise ValueError(f"Modo desconhecido: {mode!r}. Use 'max_confidence' ou 'entropy'.")

    calm_mask = (1.0 - edges).to(dtype=uncertainty.dtype)
    calm_mask = uncertainty * calm_mask

    #img = calm_mask[0].detach().cpu().numpy() 
    #plt.imsave('vis_output/calm_mask.png', img, cmap='gray')
    return calm_mask

class CRFasRNN_Fixed(nn.Module):
    def __init__(self, num_classes, num_iterations=10,
                 gaussian_ksize=5, gaussian_sigma=0.5,
                 bilateral_ksize=5, bilateral_sigma_color=0.5, bilateral_sigma_space=5.0,
                 edge_ksize=5, edge_sigma_color=0.5, edge_sigma_space=5.0):
                 
        
        """
        CRF as RNN sem parâmetros aprendíveis.
        - num_classes: número de classes C.
        - num_iterations: iterações de refinamento.
        - Parâmetros de filtro fixos (tamanho e sigma).
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        # --- Filtro Gaussiano Espacial fixo ---
        # Criar kernel Gaussiano 2D (Gaussian blur) de dimensão gaussian_ksize.
        # Usamos padding = ksize//2 para manter o tamanho.
        # O kernel é o mesmo para todos os canais, aplicável via conv2d com groups=num_classes.
        '''
        grid = torch.arange(gaussian_ksize) - (gaussian_ksize - 1) / 2
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * gaussian_sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        # Inicializa conv separada para cada canal
        kernel = gaussian_kernel.view(1, 1, gaussian_ksize, gaussian_ksize)
        self.register_buffer('gaussian_kernel', kernel)  # formato [1,1,k,k]
        '''
        # --- Filtro Gaussiano Espacial via Kornia ---
        # Cria o módulo GaussianBlur2d que aplica depth‑wise blur em q ([B,C,H,W])
        self.gaussian_blur = KF.GaussianBlur2d(
            kernel_size=(gaussian_ksize, gaussian_ksize),
            sigma=(gaussian_sigma, gaussian_sigma),
            border_type='reflect',
            separable=True
        )

        # --- Matrizes de Compatibilidade fixas (1x1 conv) ---
        # Usamos o modelo de Potts: 0 na diagonal, -1 nas off-diagonais.
        compat = -torch.ones((num_classes, num_classes))
        compat.fill_diagonal_(0)
        
        # Formatar para conv2d 1x1: [C_out, C_in, 1, 1]
        self.register_buffer('compat_spatial', compat.view(num_classes, num_classes, 1, 1))
        self.register_buffer('compat_rgb', compat.view(num_classes, num_classes, 1, 1))
        self.register_buffer('compat_edge', compat.view(num_classes, num_classes, 1, 1))
        self.register_buffer('bilateral_sigma_color', torch.tensor(bilateral_sigma_color))
        self.register_buffer('bilateral_sigma_space', torch.tensor(bilateral_sigma_space))
        self.register_buffer('edge_sigma_color',     torch.tensor(edge_sigma_color))
        self.register_buffer('edge_sigma_space',     torch.tensor(edge_sigma_space))
        self.bilateral_ksize = bilateral_ksize
        self.edge_ksize = edge_ksize

    def forward(self, unary_logits, image, edges):
        """
        unary_logits: Tensor [B, C, H, W] com logit de cada classe.
        image: Tensor [B, 3, H, W] da imagem (supõe-range [0,1]).
        Retorna mapa de probabilidades refinado [B, C, H, W].
        """
        B, C, H, W = unary_logits.shape
        q = F.softmax(unary_logits, dim=1)  # [B, C, H, W]
        #print(f"q_softmax.mean={q.mean()}, q_softmax.std={q.std()}")#2.440253496170044
        
        edges = edges.unsqueeze(1)#.expand(-1, C, -1, -1)
        
        sigma_color_rgb  = self.bilateral_sigma_color .view(1).expand(B)
        sigma_color_edge = self.edge_sigma_color      .view(1).expand(B)
        sigma_space_rgb  = self.bilateral_sigma_space.view(1, 1).expand(B, 2)
        sigma_space_edge = self.edge_sigma_space    .view(1, 1).expand(B, 2)
        
        #gaussian_weight = self.gaussian_kernel.repeat(C, 1, 1, 1)
        #pad_g = self.gaussian_kernel.size(2) // 2
        
        for _ in range(self.num_iterations):
            spatial_filtered = self.gaussian_blur(q) 
            rgb_filtered = kornia.filters.joint_bilateral_blur(
                q, image, (self.bilateral_ksize, self.bilateral_ksize),
                sigma_color=sigma_color_rgb, sigma_space=sigma_space_rgb,
                color_distance_type='l1', border_type='reflect',
            )
            edge_filtered = kornia.filters.joint_bilateral_blur(
                q, edges, (self.edge_ksize, self.edge_ksize),
                sigma_color=sigma_color_edge, sigma_space=sigma_space_edge,
                color_distance_type='l1', border_type='reflect',
            )
            pairwise_spatial = F.conv2d(spatial_filtered, self.compat_spatial)
            pairwise_rgb     = F.conv2d(rgb_filtered,     self.compat_rgb)
            pairwise_edge    = F.conv2d(edge_filtered,    self.compat_edge)
            
            #w_spatial = 0.8
            #w_rgb = 0.8  
            #w_edge = 0.8
            #pairwise = w_spatial * pairwise_spatial + w_rgb * pairwise_rgb + w_edge * pairwise_edge
            pairwise = 1.0*(pairwise_spatial + pairwise_rgb + pairwise_edge)#
            #print(f"pairwise.mean={pairwise.mean()}, pairwise.std={pairwise.std()}")
            
            q = F.softmax(unary_logits - pairwise, dim=1)
            #print(f"q_pairwise.mean={q.mean()}, q_pairwise.std={q.std()}")
        
        '''
        print(f"edge_filtered.shape={edge_filtered.shape}")
        img = spatial_filtered[0, 0].detach().cpu().numpy() 
        plt.imsave('vis_output/spatial_0.png', img, cmap='gray')
        img = rgb_filtered[0, 0].detach().cpu().numpy() 
        plt.imsave('vis_output/rgb_0.png', img, cmap='gray')
        img = edge_filtered[0, 0].detach().cpu().numpy() 
        plt.imsave('vis_output/edges_0.png', img, cmap='gray')
        '''

        return q


# Paleta de cores da segmentação
palette = {
    0: (255, 0, 0),
    1: (38, 115, 0),
    2: (0, 0, 0),
    3: (133, 199, 126),
    4: (255, 255, 0),
    5: (128, 128, 128),
    6: (139, 69, 19),
    7: (84, 117, 168)
}

def apply_palette(mask, palette):
    """Converte uma máscara (H,W) para (H,W,3) com base na paleta."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        color_mask[mask == cls] = color
    return color_mask

def visualize_first_and_save(inputs, edges, outputs, crf_out, save_path="vis_output", idx=0):
    """
    Salva uma figura 2x2 com:
    - RGB de entrada
    - bordas (grayscale)
    - saída da CNN (argmax colorida)
    - saída do CRF (argmax colorida)
    """
    # Extrai o primeiro item do batch
    inp  = inputs[0].detach().cpu().permute(1,2,0).numpy()      # [H,W,3]
    edg  = edges[0].detach().cpu().numpy()                      # [H,W]
    pred = outputs.argmax(dim=1)[0].detach().cpu().numpy()      # [H,W]
    crf_pred = crf_out.argmax(dim=1)[0].detach().cpu().numpy()  # [H,W]

    # Aplica as cores nas máscaras
    pred_color = apply_palette(pred, palette)
    crf_color  = apply_palette(crf_pred, palette)

    # Cria figura 2x2
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    axs[0].imshow(inp)
    axs[0].set_title("Input RGB")
    axs[1].imshow(edg, cmap="gray")
    axs[1].set_title("GT Edges")
    axs[2].imshow(pred_color)
    axs[2].set_title("CNN Pred")
    axs[3].imshow(crf_color)
    axs[3].set_title("CRF Refined")

    for ax in axs:
        ax.axis("off")

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/vis_batch_{idx:03d}.png", bbox_inches='tight')
    plt.close(fig)

def gaussian_kernel(kernel_size: int, sigma: float, device):
    """Cria um kernel 2D gaussiano para convolução."""
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


class CRFasRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_iterations: int = 10,
        gaussian_ks: int = 5,
        gaussian_sigma: float = 0.1,# 5.0, 0.5, 0.05, 0.005
        # bilateral RGB
        bilateral_ks: int = 5,
        bilateral_spatial_sigma: float = 5.0,
        bilateral_color_sigma: float = 0.1, # 1.0 0.1, 0.01, 0.001,
        # bilateral Edge
        edge_ks: int = 5,
        edge_spatial_sigma: float = 5.0,
        edge_intensity_sigma: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        self.gaussian_ks = gaussian_ks
        self.gaussian_sigma = gaussian_sigma

        self.bilateral_ks = bilateral_ks
        self.bilateral_spatial_sigma = bilateral_spatial_sigma
        self.bilateral_color_sigma = bilateral_color_sigma

        self.edge_ks = edge_ks
        self.edge_spatial_sigma = edge_spatial_sigma
        self.edge_intensity_sigma = edge_intensity_sigma

        # compatibilidade aprendível C×C no modo Potts
        kappa = 1.0  # fator de penalização; ajuste conforme necessidade
        init = -kappa * torch.eye(num_classes, num_classes, device=self.device)
        self.compat_mat = nn.Parameter(init)

    def forward(self, logits: torch.Tensor, img: torch.Tensor, gt_edges: torch.Tensor):
        """
        logits:   [B, C, H, W]
        img:      [B, 3, H, W]   — RGB normalizado em [0,1]
        gt_edges: [B,  H, W]     — mapa binário/probabilístico de bordas

        Note: q é inicializado fora do loop para representar a distribuição unária inicial
        e então iterativamente refinado dentro do loop de mean‑field.
        """
        B, C, H, W = logits.shape

        q = F.softmax(logits, dim=1)  # [B,C,H,W]

        # pré‑cria kernels no device
        gk = gaussian_kernel(self.gaussian_ks, self.gaussian_sigma, self.device) \
            .view(1, 1, self.gaussian_ks, self.gaussian_ks)
        pad_g = self.gaussian_ks // 2
        pad_b = self.bilateral_ks // 2
        
        K2 = self.bilateral_ks ** 2
        spatial_w = gaussian_kernel(
        self.bilateral_ks, self.bilateral_spatial_sigma, self.device
        ).contiguous().view(1, 1, K2, 1)

        ek = gaussian_kernel(self.edge_ks, self.edge_spatial_sigma, self.device) \
            .view(1, 1, self.edge_ks, self.edge_ks)
        pad_e = self.edge_ks // 2

        for it in range(self.num_iterations):
            # 1) Gaussian spatial
            spatial = F.conv2d(q.reshape(B * C, 1, H, W), gk, padding=pad_g)
            spatial = spatial.reshape(B, C, H, W)

            # 2) Bilateral RGB (unfold)
            q_unf = F.unfold(q, kernel_size=self.bilateral_ks, padding=pad_b)
            img_unf = F.unfold(img, kernel_size=self.bilateral_ks, padding=pad_b)
            _, _, L = q_unf.shape

            q_unf = q_unf.view(B, C, K2, L)
            img_unf = img_unf.view(B, 3, K2, L)
            center = img.unsqueeze(2).contiguous().view(B, 3, 1, H * W)
            diff = img_unf - center
            color_w = torch.exp(-(diff**2).sum(1, keepdim=True) /
                                (2 * self.bilateral_color_sigma**2))
            w = spatial_w * color_w
            bilateral_rgb = (q_unf * w).sum(2).contiguous().view(B, C, H, W)

            # 3) Bilateral Edge
            e_map = gt_edges.unsqueeze(1)  # [B,1,H,W]
            e_feat = F.conv2d(e_map, ek, padding=pad_e)  # [B,1,H,W]
            bilateral_edge = q * e_feat  # broadcast em C

            pairwise = 1.0 * (spatial + bilateral_rgb)# + bilateral_edge
            pairwise = torch.einsum('ij,bjpq->bipq', self.compat_mat, pairwise)

            q = F.softmax(logits - pairwise, dim=1)

        # salvar visualizações
        '''
        out_spatial = spatial[0, 0].detach().cpu().numpy()
        plt.imsave('vis_output/spatial.png', out_spatial, cmap='gray')
        out_rgb = bilateral_rgb[0, 0].detach().cpu().numpy()
        plt.imsave('vis_output/rgb.png', out_rgb, cmap='gray')
        out_edges = bilateral_edge[0, 0].detach().cpu().numpy()
        plt.imsave('vis_output/edges.png', out_edges, cmap='gray')
        '''

        return q

def kl_divergence_gpu(
    outputs: torch.Tensor,
    crf_outputs: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    
    #print(f"outputs.max() = {outputs.max()}")
    outputs = F.log_softmax(outputs, dim=1)
    #print(f"outputs.max() = {outputs.max()}")
    crf_outputs = crf_outputs.clamp(min=1e-8)
    
    kl_loss = F.kl_div(outputs, crf_outputs, reduction=reduction)
    #print(f"kl_loss.shape = {kl_loss.shape}")
    kl_loss = kl_loss.sum(dim=1)
    #print(f"kl_loss.shape = {kl_loss.shape}")
    #kl_loss = kl_loss.mean()
    #print(f"kl_loss = {kl_loss}")
    '''
    # supondo que `outputs` já seja softmax (p) e `crf_outputs` continue sendo softmax (q)
    log_q = torch.log(crf_outputs)           # log q_{ij,c}
    p = outputs.softmax(dim=1)               # p_{ij,c}
    
    kl_loss = F.kl_div(log_q, p, reduction='none')
    # = p * (log p − log q) → D_KL(P||Q) por pixel
    kl_loss = kl_loss.sum(dim=1)#.mean()
    '''
    return kl_loss



class Trainer():
    
    def __init__(self, net, loader, params, scheduler = True, cbkp = None,):
        
        self.net = net
        self.loader = loader
        self.params = params
        self.iter_ = 0
        
        self.width = 224
        self.height = 224
        
        self.epoch_loss = []
        self.epoch_val_loss = []
        self.accuracies = []
        self.epoch_acc = []
        self.epoch_val_acc = []
        self.epoch_f1 = []
        self.epoch_val_f1 = []
        self.epoch_mcc = []
        self.epoch_val_mcc = []
        self.epoch_iou = []
        self.epoch_val_iou = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.losses = np.zeros(1000000)
        self.mean_losses = np.zeros(100000000)
        self.print_each = params['print_each'] or 100 
        
        # # Weights for class balancing
        #self.weight_cls = self.prepare([self.params['weights']])
        self.CE = nn.CrossEntropyLoss(reduce=None, reduction="none")#, weight=self.weight_cls[0]
        self.FL = FocalLoss(mode='multiclass', alpha=self.params['loss']['params']['alpha'], gamma=self.params['loss']['params']['gamma'], reduction='none')
        self.DI = DiceLoss(
            mode='multiclass',
            classes=8,  # ou lista de índices de classes a considerar
            log_loss=False,
            from_logits=True,  # True se sua rede não aplica softmax na saída
        )
        self.JL = JaccardLoss(
            mode='multiclass',
            classes=8,  # ou lista de índices de classes a considerar
            log_loss=False,
            from_logits=True,  # True se sua rede não aplica softmax na saída
        )
        self.TV = TverskyLoss(
            mode='multiclass',
            classes=8,  # ou lista de índices de classes a considerar
            log_loss=False,
            from_logits=True,  # True se sua rede não aplica softmax na saída
            alpha=1.0, beta=0.5
        )
        
        # Define an id to a trained model. Use the number of seconds since 1970
        time_ = str(time.time())
        time_ = time_.replace(".", "")
        self.model_id = time_

        # Create optimizer
        self.optimizer = make_optimizer(self.params['optimizer_params'], self.net)
        # Create scheduler
        self.scheduler = make_scheduler(self.params['lrs_params'], self.optimizer) if scheduler else None

        self.last_epoch = 0

        # Load a previously model if it exists
        if cbkp is not None:
            self.load(cbkp)

        Path(os.path.join(self.params['results_folder'])).mkdir(parents=True, exist_ok=True)
        
    def plot_metrics(self, results_folder):
        # Ensure the output directory exists
        os.makedirs(results_folder, exist_ok=True)
    
        # Determine number of recorded epochs
        n_epochs = len(self.epoch_acc)
        if n_epochs == 0:
            # Nothing to plot yet
            return
        
        plt.figure(figsize=(8, 6))
        
        # Define cores base
        color_acc = 'tab:blue'
        color_f1  = 'tab:orange'
        color_iou  = 'tab:red'
        color_mcc  = 'tab:green'
        
        # Plot das curvas de Accuracy
        plt.plot(self.epoch_acc,       label='Train Accuracy',      color=color_acc, linestyle='-')
        plt.plot(self.epoch_val_acc,   label='Val Accuracy',        color=color_acc, linestyle='--')
        
        # Plot das curvas de F1-score
        plt.plot(self.epoch_f1,        label='Train F1-score',      color=color_f1, linestyle='-')
        plt.plot(self.epoch_val_f1,    label='Val F1-score',        color=color_f1, linestyle='--')
        
        # Plot das curvas de F1-score
        plt.plot(self.epoch_iou,        label='Train IoU',      color=color_iou, linestyle='-')
        plt.plot(self.epoch_val_iou,    label='Val IoU',        color=color_iou, linestyle='--')
        
        # Plot das curvas de F1-score
        plt.plot(self.epoch_mcc,        label='Train MCC',      color=color_mcc, linestyle='-')
        plt.plot(self.epoch_val_mcc,    label='Val MCC',        color=color_mcc, linestyle='--')
        
        # Labels e título
        plt.xlabel('Epoch')
        plt.ylabel('Métrica')
        plt.title('Treinamento vs Validação')
        
        # Legenda e grid
        plt.legend()
        plt.grid(True)
        
        # Ajusta layout e salva
        plt.tight_layout()
        save_path = os.path.join(results_folder, 'metrics_all_in_one.png')
        plt.savefig(save_path)
        plt.close()
    
        # X-axis values based on actual recorded epochs
        epochs = list(range(1, n_epochs + 1))
        '''
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        # Plot accuracy curves on the left
        ax1.plot(epochs, self.epoch_acc, label='Train Accuracy')
        ax1.plot(epochs, self.epoch_val_acc, label='Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True)
    
        # Plot F1-score curves on the right
        ax2.plot(epochs, self.epoch_f1, label='Train F1-score')
        ax2.plot(epochs, self.epoch_val_f1, label='Validation F1-score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Training vs Validation F1-score')
        ax2.legend()
        ax2.grid(True)
    
        # Adjust layout and save the figure
        plt.tight_layout()
        save_path = os.path.join(results_folder, f'metrics_epoch.png')
        plt.savefig(save_path)
        plt.close(fig)
        '''    
    def load(self, path):
        
        # Check if model file exists
        assert os.path.exists(path), "{} cant be opened".format(path)
        
        checkpoint = torch.load(path)

        try:
            self.last_epoch = checkpoint['epoch']
            self.model_id = checkpoint['model_id']
            self.losses = checkpoint['losses']
            self.mean_losses = checkpoint['mean_losses']
            self.iter_ = checkpoint['iter_']
            
            # Backup de metricas
            self.epoch_loss = checkpoint['epoch_loss']
            self.epoch_val_loss = checkpoint['epoch_val_loss']
            self.epoch_acc = checkpoint['epoch_acc']
            self.epoch_val_acc = checkpoint['epoch_val_acc']
            self.epoch_f1 = checkpoint['epoch_f1']
            self.epoch_val_f1 = checkpoint['epoch_val_f1']
            self.epoch_mcc = checkpoint['epoch_mcc']
            self.epoch_val_mcc = checkpoint['epoch_val_mcc']
            self.epoch_iou = checkpoint['epoch_iou']
            self.epoch_val_iou = checkpoint['epoch_val_iou']
            
            self.accuracies = checkpoint['accuracies']
            # self.acc_ = checkpoint['acc_']
        except KeyError as e:
            print(e)
            pass
            
        # Load model and optimizer params
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        #if self.scheduler is not None:
        #    for _ in range(0, self.last_epoch): self.scheduler.step()

    def save(self, path = None):

        if path is None:
            # path = './{}_model_final.pth.tar'.format(self.model_id)
            path = os.path.join(self.params['results_folder'], f"{self.model_id}_model_final.pth.tar")

        # Save current loss, epoch, model weights and optimizer params
        torch.save({
            'epoch': self.last_epoch,
            'losses': self.losses,
            'mean_losses': self.mean_losses,
            
            # Backup de metricas
            'epoch_loss': self.epoch_loss,
            'epoch_val_loss': self.epoch_val_loss,
            'epoch_acc': self.epoch_acc,
            'epoch_val_acc': self.epoch_val_acc,
            'epoch_f1': self.epoch_f1,
            'epoch_val_f1': self.epoch_val_f1,
            'epoch_mcc': self.epoch_mcc,
            'epoch_val_mcc': self.epoch_val_mcc,
            'epoch_iou': self.epoch_iou,
            'epoch_val_iou': self.epoch_val_iou,
            
            'accuracies': self.accuracies,
            #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            'iter_': self.iter_,
            # 'acc_': self.acc_,
            'model_id': self.model_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def prepare(self, l, non_blocking=True):
        def _prepare(tensor: torch.Tensor):
            if self.params['precision'] == 'half': tensor = tensor.half() # Convert to half precision
            # return tensor.to(device, non_blocking=non_blocking)
            return tensor.to(self.device, non_blocking=non_blocking)
        return [_prepare(_l) for _l in l]
        
    def test_mc_dropout(self, test_loader=None, stride=None, window_size=None, batch_size=None,
                        mc_runs=25, mc_dropout_rates: Optional[Sequence[float]] = None,
                        seed_base: Optional[int] = None, return_detailed: bool = False):
        """
        MC Dropout test (retorna all_preds = predições da melhor run para cada imagem).
        - return_detailed: se True retorna (all_preds, details_dict), se False retorna all_preds.
        """
        import torch
        import torch.nn as nn
        import numpy as np
        from torch.nn import functional as F
        from tqdm import tqdm
        from scipy.special import softmax
        import random
    
        # carregamento e asserts
        test_ld = test_loader if test_loader is not None else self.loader['test']
        assert test_ld is not None, "Test_loader can't be None"
    
        test_stride = stride if stride is not None else self.params['stride']
        assert test_stride is not None, "Stride not set"
    
        test_ws = window_size if window_size is not None else self.params['window_size']
        assert test_ws is not None, "Window size not set"
    
        bs = batch_size if batch_size is not None else self.params.get('bs', 1)
        bs = bs if bs is not None else 1
    
        input_ids, label_ids, _ = test_ld.dataset.get_dataset()
        input_ids = list(input_ids)
        label_ids = list(label_ids)
        n_images = len(input_ids)
        n_classes = self.params['n_classes']
    
        # preparar dropout rates
        if mc_dropout_rates is not None:
            if isinstance(mc_dropout_rates, (list, tuple, np.ndarray)):
                assert len(mc_dropout_rates) == mc_runs, "mc_dropout_rates length must match mc_runs"
                dropout_rates_list = list(mc_dropout_rates)
            else:
                dropout_rates_list = [float(mc_dropout_rates)] * mc_runs
        else:
            dropout_rates_list = [None] * mc_runs
    
        # helpers
        def set_mc_dropout_mode(model, enable: bool, p: Optional[float] = None):
            model.eval()
            for m in model.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    if enable:
                        m.train()
                    else:
                        m.eval()
                    if (p is not None) and hasattr(m, 'p'):
                        m.p = float(p)
    
        def infer_image_once(image_np):
            H, W = image_np.shape[:2]
            pred_proba = np.zeros((H, W, n_classes), dtype=np.float32)
            #count_map = np.zeros((H, W), dtype=np.float32)
    
            sliding_iter = sliding_window(image_np, stride=test_stride, window_size=test_ws)
            total = max(1, count_sliding_window(image_np, stride=test_stride, window_size=test_ws) // bs)
            for coords_batch in tqdm(grouper(bs, sliding_iter), total=total, leave=False):
                coords = [c for c in coords_batch if c is not None]
                if len(coords) == 0:
                    continue
    
                patches = []
                for x, y, w, h in coords:
                    patch = np.copy(image_np[x:x+w, y:y+h])
                    patch = patch.transpose((2,0,1))
                    patches.append(patch)
                inp_np = np.asarray(patches, dtype=np.float32)
                inp = torch.from_numpy(inp_np).float().to(self.device)
                with torch.no_grad():
                    outs = self.net(inp)  # (bs, n_classes, h, w)
                outs = outs.cpu().numpy()
                
                for i, (x, y, w, h) in enumerate(coords):
                    patch_logits = np.transpose(outs[i], (1,2,0))   # (h, w, n_classes)
                    pred_proba[x:x+w, y:y+h, :] += patch_logits
    
            pred_proba = softmax(pred_proba, axis=-1)
            pred_class = np.argmax(pred_proba, axis=-1).astype(np.uint8)
            return pred_proba, pred_class
    
        # acumulos
        preds_per_image = [list() for _ in range(n_images)]   # preds_per_image[img_idx][run_idx]
        proba_sums = [None for _ in range(n_images)]         # soma das probs por imagem ao longo das runs
    
        per_run_acc = []
        per_run_f1  = []
        per_run_iou = []
        per_run_mcc = []
        per_run_details = []
    
        # loop runs (fixa dropout por run e processa todo o conjunto)
        for run_idx in range(mc_runs):
            # seed por run (opcional)
            if seed_base is not None:
                seed = int(seed_base) + int(run_idx)
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
            p = dropout_rates_list[run_idx]
            set_mc_dropout_mode(self.net, enable=True, p=p)
    
            # coletores para métricas agregadas desta run
            all_preds_flat_list = []
            all_gts_flat_list = []
    
            # inferir TODO o conjunto
            for img_idx, (image_path, label_path) in enumerate(tqdm(zip(input_ids, label_ids),
                                                                     total=n_images, desc=f"run {run_idx+1}/{mc_runs}",
                                                                     leave=False)):
                img = (io.imread(image_path)[:,:,:3].astype('float32') / 255.0)
                gt_e = convert_from_color(io.imread(label_path)[:,:,:3])
    
                pred_proba_run, pred_class_run = infer_image_once(img)
    
                # acumula soma de probabilidades por imagem
                if proba_sums[img_idx] is None:
                    proba_sums[img_idx] = pred_proba_run.astype(np.float32)
                else:
                    proba_sums[img_idx] += pred_proba_run.astype(np.float32)
    
                preds_per_image[img_idx].append(pred_class_run)
    
                all_preds_flat_list.append(pred_class_run.reshape(-1))
                all_gts_flat_list.append(gt_e.reshape(-1))
    
            # compute aggregated metrics for this run (over all images)
            preds_all = np.concatenate(all_preds_flat_list, axis=0).astype(np.int64)
            gts_all   = np.concatenate(all_gts_flat_list, axis=0).astype(np.int64)
    
            # use torchmetrics on CPU
            import torchmetrics
            device_metrics = 'cpu'
            acc_metric = torchmetrics.Accuracy(num_classes=n_classes, average='macro', task='multiclass').to(device_metrics)
            f1_metric  = torchmetrics.F1Score(num_classes=n_classes, average='macro', task='multiclass').to(device_metrics)
            iou_metric = torchmetrics.JaccardIndex(num_classes=n_classes, average='macro', task='multiclass').to(device_metrics)
            mcc_metric = torchmetrics.MatthewsCorrCoef(num_classes=n_classes, task='multiclass').to(device_metrics)
    
            pred_tensor_cpu = torch.tensor(preds_all, device=device_metrics)
            gt_tensor_cpu   = torch.tensor(gts_all, device=device_metrics)
    
            acc_metric.update(pred_tensor_cpu, gt_tensor_cpu)
            f1_metric.update(pred_tensor_cpu, gt_tensor_cpu)
            iou_metric.update(pred_tensor_cpu, gt_tensor_cpu)
            mcc_metric.update(pred_tensor_cpu, gt_tensor_cpu)
    
            acc_val = float(acc_metric.compute().item()) * 100.0
            f1_val  = float(f1_metric.compute().item()) * 100.0
            iou_val = float(iou_metric.compute().item()) * 100.0
            mcc_val = float(mcc_metric.compute().item()) * 100.0
    
            per_run_acc.append(acc_val)
            per_run_f1.append(f1_val)
            per_run_iou.append(iou_val)
            per_run_mcc.append(mcc_val)
    
            per_run_details.append({
                'run_idx': run_idx, 'dropout_p': p,
                'acc': acc_val, 'f1': f1_val, 'iou': iou_val, 'mcc': mcc_val
            })
    
            # desligar Dropout modules para segurança (opcional)
            set_mc_dropout_mode(self.net, enable=False)
    
        # média de probabilidades por imagem (sobre runs)
        mean_preds_by_image = []
        for img_idx in range(n_images):
            proba_sum = proba_sums[img_idx]
            assert proba_sum is not None, "Alguma imagem não foi processada em nenhuma run."
            mean_proba_img = proba_sum / float(mc_runs)
            mean_pred_img = np.argmax(mean_proba_img, axis=-1).astype(np.uint8)
            mean_preds_by_image.append(mean_pred_img)
    
        # escolher melhor run (maior IoU agregado por run)
        best_run_idx = int(np.argmax(per_run_iou))
        # construir all_preds no formato "uma predição por imagem do best run"
        best_run_preds = [preds_per_image[i][best_run_idx] for i in range(n_images)]
        all_preds = np.array(best_run_preds)  # shape (n_images, H, W) — mesma semântica do seu código original
    
        # imprimir resumo compacto
        print("==== Estatísticas por run (agregadas sobre todas as imagens) ====")
        for d in per_run_details:
            print(f"Run {d['run_idx']:02d} | IoU={d['iou']:.4f} | F1={d['f1']:.4f} | Acc={d['acc']:.4f} | MCC={d['mcc']:.4f}")# p={d['dropout_p']} |
        print(f"\nMelhor run (maior IoU agregado): run {best_run_idx} (IoU={per_run_iou[best_run_idx]:.4f})")
     
        ious = np.array([d['iou'] for d in per_run_details])
        f1s  = np.array([d['f1']  for d in per_run_details])
        accs = np.array([d['acc'] for d in per_run_details])
        mccs = np.array([d['mcc'] for d in per_run_details])
        
        print(f"\nMédias ± Desvios -> IoU: {ious.mean():.2f}±{ious.std():.2f} | "
              f"F1: {f1s.mean():.2f}±{f1s.std():.2f} | "
              f"Acc: {accs.mean():.2f}±{accs.std():.2f} | "
              f"MCC: {mccs.mean():.2f}±{mccs.std():.2f}")
        # preparar objeto de retorno detalhado
        stats = {
            'per_run': {
                'acc': per_run_acc, 'f1': per_run_f1, 'iou': per_run_iou, 'mcc': per_run_mcc
            },
            'per_run_details': per_run_details,
            'mean_preds_by_image_sample': len(mean_preds_by_image)
        }
        
        accuracy = metrics(
            predictions=np.concatenate([p.ravel() for p in all_preds]), 
            gts=np.concatenate([p.ravel() for p in gts_all]).ravel(),
            label_values=self.params['classes'],
            all=all,
            filepath=self.params['results_folder']
        )
    
        if return_detailed:
            return all_preds, {'stats': stats, 'mean_preds_by_image': mean_preds_by_image, 'preds_per_image_all_runs': preds_per_image}
        else:
            return all_preds


    def test(self, test_loader = None, stride = None, window_size = None, batch_size = None, all=False):
        test_ld = test_loader if test_loader is not None else self.loader['test']
        assert test_ld is not None, "Test_loader can't be None"

        test_stride = stride if stride is not None else self.params['stride']
        assert test_stride is not None, "Stride not set"

        test_ws = window_size if window_size is not None else self.params['window_size']
        assert test_ws is not None, "Window size not set"

        bs = batch_size if batch_size is not None else self.params['bs']
        bs = bs if bs is not None else 1

        input_ids, label_ids, _ = test_ld.dataset.get_dataset()

        test_images = (1 / 255 * np.asarray(io.imread(id)[:,:,:3], dtype='float32') for id in input_ids)
        test_labels = (np.asarray(io.imread(id)[:,:,:3], dtype='uint8') for id in label_ids)
        eroded_labels = (convert_from_color(io.imread(id)[:,:,:3]) for id in label_ids)
        
        all_preds = []
        all_gts = []
        
        acc_metric = Accuracy(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        f1_metric = F1Score(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        iou_metric = JaccardIndex(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        mcc_metric = MatthewsCorrCoef(num_classes=self.params['n_classes'], task='multiclass').to(self.device)
        
        self.net.eval()
        
        for img, gt, gt_e, image_path in tqdm(zip(test_images, test_labels, eroded_labels, input_ids), total=len(input_ids), leave=False):
            filepath = os.path.split(image_path)[1].split('.')[0]
            Path(os.path.join(self.params['results_folder'], 'inference', filepath)).mkdir(parents=True, exist_ok=True)
            
            pred = np.zeros(img.shape[:2] + (self.params['n_classes'],))

            total = count_sliding_window(img, stride=test_stride, window_size=test_ws) // bs
            for i, coords in enumerate(tqdm(grouper(bs, sliding_window(img, stride=test_stride, window_size=test_ws)), total=total, leave=False)):     
                with torch.no_grad():
                    image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())
                    lbl_patches = [np.copy(gt[x:x+w, y:y+h]) for x,y,w,h in coords]
                
                outs = self.net(image_patches)
                outs = outs.data.cpu().numpy()
                
                for out, (x, y, w, h) in zip(outs, coords):
                    pred[x:x+w, y:y+h] += out.transpose((1,2,0))
                    
            soft = np.argmax(pred, axis=-1)

            all_preds.append(soft)
            all_gts.append(gt_e)

            pred_tensor = torch.tensor(soft.flatten(), device=self.device)
            gt_tensor = torch.tensor(gt_e.flatten(), device=self.device)
    
            acc_metric.update(pred_tensor, gt_tensor)
            f1_metric.update(pred_tensor, gt_tensor)
            iou_metric.update(pred_tensor, gt_tensor)
            mcc_metric.update(pred_tensor, gt_tensor)
        
        acc = acc_metric.compute().item()*100.0
        f1 = f1_metric.compute().item()*100.0
        iou = iou_metric.compute().item()*100.0
        mcc = mcc_metric.compute().item()*100.0
    
        print(f"Accuracy (macro): {acc:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"IoU (macro): {iou:.4f}")
        print(f"MCC: {mcc:.4f}")
        
        accuracy = metrics(
            predictions=np.concatenate([p.ravel() for p in all_preds]), 
            gts=np.concatenate([p.ravel() for p in all_gts]).ravel(),
            label_values=self.params['classes'],
            all=all,
            filepath=self.params['results_folder']
        )

        if all:
            save_test(acc=accuracy, all_preds=all_preds, all_gts=all_gts, 
                      path=os.path.join(self.params['results_folder'], 'test_result.npz'))
            return all_preds#accuracy, , all_gts
        else:
            return accuracy
        

    def train(self):
        train_running_loss = 0.0
        train_running_correct = 0
        train_running_f1 = 0
        train_running_f1G = 0

        self.last_epoch = self.scheduler.last_epoch + 1
        pbar = tqdm(self.loader['train'])
        
        acc_metric = Accuracy(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        f1_metric = F1Score(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        iou_metric = JaccardIndex(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        mcc_metric = MatthewsCorrCoef(num_classes=self.params['n_classes'], task='multiclass').to(self.device)
        
        
        self.net.train()
        counter = 0
        for batch_id, (inputs, labels) in enumerate(pbar):   
            inputs, labels = self.prepare([inputs, labels])
            counter += labels.size(0)
            inputs = inputs[:,:3,:,:]
            
            self.optimizer.zero_grad()
            outputs = self.net(inputs)              # [B,C,H,W]
            
            probs = F.softmax(outputs, dim=1)
            
            #loss = self.FL(probs, labels).mean()
            labels = labels.long()
            #loss = self.CE(outputs, labels).mean()
            #loss = self.DI(outputs, labels).mean()
            #loss = self.JL(outputs, labels).mean()
            loss = self.TV(outputs, labels)#.mean()
            
            loss.backward()
            self.optimizer.step()
            
            max_values, armax = torch.max(probs.data, 1)
            
            acc_metric.update(armax, labels)
            f1_metric.update(armax, labels)
            mcc_metric.update(armax, labels)
            iou_metric.update(armax, labels)
            
            acc = 100.0*acc_metric.compute().item()
            f1 = 100.0*f1_metric.compute().item()
            mcc = 100.0*mcc_metric.compute().item()
            iou = 100.0*iou_metric.compute().item()
            train_running_loss += loss.item()
            
            pbar.set_postfix({
                'Epoch': self.last_epoch, 
                'Acc': acc,
                'F1': f1,
                'MCC': mcc,
                'IoU': iou,
                'Loss': train_running_loss/(batch_id+1),
            })

            self.iter_ += 1
            del(inputs, labels, loss)
            
        return acc, f1, mcc, iou
    
    def validate(self, stride=32, window_size=(224,224), batch_size=None, all=False):
        start_time = time.time() 
        val_ld = self.loader.get('val', None)
        assert val_ld is not None

        input_ids, label_ids, _ = val_ld.dataset.get_dataset()

        bs = batch_size or self.params.get('bs', 1)
        self.net.eval()

        acc_metric = Accuracy(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        f1_metric = F1Score(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        iou_metric = JaccardIndex(num_classes=self.params['n_classes'], average='macro', task='multiclass').to(self.device)
        mcc_metric = MatthewsCorrCoef(num_classes=self.params['n_classes'], task='multiclass').to(self.device)

        acc_va = 0.0
        train_running_f1G=0.0
        c = 0
        with torch.no_grad():
            for img_path, lbl_path in tqdm(zip(input_ids, label_ids),
                                           total=len(input_ids),
                                           desc="Validação", leave=False):
                img = io.imread(img_path)[..., :3] 
                img = img.astype(np.float32) / 255.0  
                img = torch.from_numpy(img).permute(2, 0, 1).to(self.device, non_blocking=True)
                
                gt_color = ioG.read_image(lbl_path)[:3].to(self.device)
                gt_e = convert_from_color_torch(gt_color) 
                _, H, W = img.shape
                
                total_patches = math.ceil(count_sliding_window_torch(img,
                                                     stride=stride,
                                                     window_size=window_size) / bs)

                gen = grouper(bs,
                              sliding_window_torch(img, stride=stride, window_size=window_size))

                pred = torch.zeros((self.params['n_classes'], H, W), device=self.device)
                count = torch.zeros((H, W), device=self.device)
                count_t = torch.ones((self.width, self.height), device=self.device)
                for coords in tqdm(gen, total=total_patches, leave=False):
                    patches, p2 = [], []
                    for x, y, h, w in coords:
                        patch = img[:, x:x+h, y:y+w]
                        patches.append(patch)
                        p2.append(gt_e[x:x+h, y:y+w])
                    batch = torch.stack(patches, dim=0)
                    outs = self.net(batch)
                    outs = F.softmax(outs, dim=1)
                    p2 = torch.stack(p2, dim=0)
                    #acc_patch_metric.update(outs, p2)
                    #f1_patch_metric.update(outs, p2)

                    for out, (x, y, h, w) in zip(outs, coords):
                        pred[:, x:x+h, y:y+w] += out
                        count[x:x+h, y:y+w] += count_t

                pred_label = pred/(count.unsqueeze(0))
                pred_label = pred_label.argmax(dim=0)
                acc_metric.update(pred_label, gt_e)
                f1_metric.update(pred_label, gt_e)
                mcc_metric.update(pred_label, gt_e)
                iou_metric.update(pred_label, gt_e)

            #acc_patch_val = acc_patch_metric.compute().item() * 100.0
            #f1_patch_val = f1_patch_metric.compute().item() * 100.0
            accuracy_val = acc_metric.compute().item() * 100.0
            f1score_val = f1_metric.compute().item() * 100.0
            mcc_val = mcc_metric.compute().item() * 100.0
            iou_val = iou_metric.compute().item() * 100.0

        elapsed_time = time.time() - start_time 

        print(f"[Validation]| ACC: {accuracy_val:.2f} | F1: {f1score_val:.2f} | MCC: {mcc_val:.2f} | IoU: {iou_val:.2f} | Elapsed time: {int(elapsed_time//60):02}:{int(elapsed_time%60):02}")# ACC_p: {acc_patch_val:.2f}  F1_p: {f1_patch_val:.2f} |

        return accuracy_val, f1score_val, mcc_val, iou_val


    def __save_plots(self):
        # loss plots
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_loss), len(self.epoch_loss)).astype(int), self.epoch_loss, 'ro-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train Epoch/Loss')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_loss_curve'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # acc plots
        fig = plt.figure()
        plt.plot(np.linspace(1, len(self.epoch_acc), len(self.epoch_acc)).astype(int), self.epoch_acc, 'bo-')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('Train Epoch/Acc')
        fig.savefig(os.path.join(self.params['results_folder'], 'train_epoch_acc_curve'), dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)