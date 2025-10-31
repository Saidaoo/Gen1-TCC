# Gen1-TCC

# 🧪 Gen1 - Comparação UNet vs DeepLabV3+

Experimento científico comparando arquiteturas de segmentação semântica com hiperparâmetros 1:1.

## 🎯 Objetivo
Comparar performance entre UNet e DeepLabV3+ mantendo TODOS os hiperparâmetros idênticos, isolando apenas a diferença arquitetural.

## 📊 Dataset
- **Classes**: 8 classes de uso do solo
- **Tamanho**: 224x224 pixels
- **Split**: 5-fold cross validation
- **Classes**: Urbano, Vegetação Densa, Sombra, Vegetação Esparsa, Agricultura, Rocha, Solo Exposto, Água

## 🏗️ Arquiteturas
- **UNet**: Encoder EfficientNet-B4 + Decoder clássico
- **DeepLabV3+**: Encoder EfficientNet-B4 + ASPP + Decoder

## ⚙️ Hiperparâmetros (1:1)
| Parâmetro | Valor |
|-----------|-------|
| Batch Size | 40 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Loss Function | Tversky Loss |
| Scheduler | Plateau |
| Input Size | 224x224 |

## 📈 Métricas
- Accuracy, F1-Score, mIoU, Matthews Correlation Coefficient (MCC)

## 🚀 Como Executar
```bash
cd src/
python main.py
