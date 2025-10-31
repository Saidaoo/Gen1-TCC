import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
from tifffile import imread
import segmentation_models_pytorch as smp
import torch.nn as nn

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

input_dir = '../../dataset_35/images'
dir_path = '20250720K4_b75_drop_Plato_flipRotColor_unet_focal_loss_RADAM'
model_path = f'output_35/{dir_path}/best_epoch.pth.tar'
output_dir = f'output_35/{dir_path}/inference_val'
selected_files_path = 'folds/fold1_images.txt'  # caminho para o .txt

# === MODELO COM DROPOUT ===
class UNetWithDropout(smp.Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, block in enumerate(self.decoder.blocks):
            block.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        return super().forward(x)

model = UNetWithDropout(
    encoder_name='efficientnet-b7',
    in_channels=3,
    classes=8,
    activation=None,
    decoder_use_norm=True,
    #decoder_attention_type='scse',
    decoder_interpolation="nearest"
)

with torch.serialization.safe_globals([torch._utils._rebuild_tensor_v2]):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

checkpoint_state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
model.load_state_dict(checkpoint_state, strict=False)
model = model.to(device)
model.eval()

# === FUNCOES DE INFERENCIA ===
def sliding_window(img, window_size=(224, 224), stride=224):
    H, W, _ = img.shape
    xs = list(range(0, H - window_size[0] + 1, stride))
    if xs[-1] != H - window_size[0]:
        xs.append(H - window_size[0])
    ys = list(range(0, W - window_size[1] + 1, stride))
    if ys[-1] != W - window_size[1]:
        ys.append(W - window_size[1])

    for x in xs:
        for y in ys:
            yield x, y, window_size[0], window_size[1]

def infer_image(image, patch_size=(224, 224), stride=224, batch_size=224):
    H, W, C = image.shape
    num_classes = 8
    prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
    count_acc = np.zeros((H, W), dtype=np.float32)

    patches = []
    coords = []
    for x, y, h, w in sliding_window(image, patch_size, stride):
        patch = image[x:x+h, y:y+w, :]
        patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        patches.append(patch_tensor)
        coords.append((x, y, h, w))

        if len(patches) == batch_size:
            batch = torch.stack(patches, dim=0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            for i, (x, y, h, w) in enumerate(coords):
                prob_acc[:, x:x+h, y:y+w] += probs[i]
                count_acc[x:x+h, y:y+w] += 1
            patches, coords = [], []

    if patches:
        batch = torch.stack(patches, dim=0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        for i, (x, y, h, w) in enumerate(coords):
            prob_acc[:, x:x+h, y:y+w] += probs[i]
            count_acc[x:x+h, y:y+w] += 1

    avg_probs = prob_acc / np.maximum(count_acc, 1e-8)
    return np.argmax(avg_probs, axis=0).astype(np.uint8)

def convert_to_rgb(pred, palette):
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for class_idx, color in palette.items():
        rgb[pred == class_idx] = color
    return rgb

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

# === CARREGAR LISTA DE IMAGENS ===
with open(selected_files_path, 'r') as f:
    selected_filenames = [line.strip() for line in f if line.strip()]

print("Imagens selecionadas:")
for name in selected_filenames:
    print(name)

os.makedirs(output_dir, exist_ok=True)

# === LOOP PRINCIPAL ===
for filename in selected_filenames:
    img_path = os.path.join(input_dir, filename)
    print(f"\nProcessando {img_path}...")

    image = imread(img_path)
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[-1] != 3:
        image = image[..., :3]

    start_time = time.time()
    pred = infer_image(image, patch_size=(224, 224), stride=64, batch_size=224)
    elapsed = time.time() - start_time
    print(f"Tempo de inferencia: {elapsed:.2f} segundos")

    rgb_img = convert_to_rgb(pred, palette)
    out_name = f"inference_tile_{os.path.splitext(filename)[0]}.png"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    print(f"Imagem salva em {out_path}")