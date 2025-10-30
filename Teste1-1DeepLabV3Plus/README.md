# Gen1-TCC

# Comparação UNet vs DeepLabV3+

Experimento comparando arquiteturas de segmentação semântica com hiperparâmetros 1:1.

## 🎯 Objetivo
Comparar performance entre UNet e DeepLabV3+ mantendo todos os hiperparâmetros idênticos.

## 📊 Dataset
- 8 classes: Urbano, Vegetação Densa, Sombra, Vegetação Esparsa, Agricultura, Rocha, Solo Exposto, Água
- 5-fold cross validation
- Input size: 224x224

## 🏗️ Arquiteturas
- **UNet**: EfficientNet-B4 encoder
- **DeepLabV3+**: EfficientNet-B4 encoder + ASPP

## ⚙️ Hiperparâmetros (1:1)
- Batch Size: 40
- Learning Rate: 1e-3
- Optimizer: Adam
- Loss: Tversky Loss
- Scheduler: Plateau

## 📈 Métricas
- Accuracy, F1-Score, mIoU, MCC
