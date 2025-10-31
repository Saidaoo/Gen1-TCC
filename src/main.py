from skimage import io
import os, time
import torch
import numpy as np
import pandas as pd
from project_utils import load_loss_weights, batch_mean_and_sd

from dataset import DatasetIcmbio
from trainer import Trainer
from models import build_model
from project_utils import (
    clear,
    convert_to_color,
    make_optimizer,
    seed_everything,
    visualize_augmentations,
)


def is_save_epoch(epoch, ignore_epoch=0):
    return (
        params["save_epoch"] is not None
        and epoch % params["save_epoch"] == 0
        and epoch != ignore_epoch
    )


class LossFN:
    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    DICE = "DICE"
    JACCARD = "JACCARD"
    TVERSKY = "TVERSKY"


class ModelChooser:
    SEGNET_MODIFICADA = "segnet_modificada"
    UNET = "unet"
    SEGFORMER = "segformer"
    DEEPLABV3PLUS = "deeplabv3plus"


class Callback:

    def __init__(self, patience=10, min_value=66):

        self.PATIENCE = patience
        self.COUNTER = 0
        self.MIN_LIMIT = min_value
        self.BEST_VALUE = 0
        self.BEST_TRAINER = []

    def patience_loss(self, epoch):
        if trainer.epoch_loss[epoch - 1] < self.BEST_VALUE:
            self.BEST_VALUE = trainer.epoch_loss[epoch - 1]
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        elif trainer.epoch_loss[epoch - 1] >= self.MIN_LIMIT:
            print(f"PATIENCE :::: Loss Too High | Skipping Save...")
            return False
        else:
            self.COUNTER += 1
            print(
                f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save..."
            )
            return False

    def patience_acc(self, epoch):
        if trainer.epoch_acc[epoch - 1] > self.BEST_VALUE:
            self.BEST_VALUE = trainer.epoch_acc[epoch - 1]
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        elif trainer.epoch_acc[epoch - 1] <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Accuracy Too low | Skipping Save...")
            return False
        else:
            self.COUNTER += 1
            print(
                f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save..."
            )
            return False

    def patience_acc_val(self, avg_acc):
        if avg_acc > self.MIN_LIMIT and avg_acc > self.BEST_VALUE:
            self.BEST_VALUE = avg_acc
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val acc < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(
                f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save..."
            )
            return False

    def patience_f1_val(self, f1):
        if f1 > self.MIN_LIMIT and f1 > self.BEST_VALUE:
            self.BEST_VALUE = f1
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val F1-Score < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(
                f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save..."
            )
            return False

    def patience_iou_val(self, iou):
        if iou > self.MIN_LIMIT and iou > self.BEST_VALUE:
            self.BEST_VALUE = iou
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val mIoU < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(
                f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save..."
            )
            return False

    def patience_loss_val(self, avg_loss):
        if avg_loss >= self.MIN_LIMIT:
            print(f"PATIENCE :::: Accuracy Too low | Skipping Save...")
            return False
        elif avg_loss < self.BEST_VALUE:
            self.BEST_VALUE = avg_loss
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        else:
            self.COUNTER += 1
            print(
                f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save..."
            )
            return False


def weights_calculator_loss(params, train_labels):
    try:
        if params["loss"]["name"] == LossFN.CROSS_ENTROPY:
            if params["loss"]["params"]["weights"] == "equal":
                params["weights"] = torch.ones(params["n_classes"])
            elif params["loss"]["params"]["weights"] == "calculate":
                if os.path.exists("./loss_weights.npy"):
                    loss_weights = load_loss_weights("./loss_weights.npy")
                    params["weights"] = torch.from_numpy(
                        loss_weights["weights"]
                    ).float()
                else:
                    import utils.weights_calculator as wc

                    loss_weights, _ = wc.WeightsCalculator(
                        train_labels, params["classes"], dev=False
                    ).calculate_and_save()
                    params["weights"] = torch.from_numpy(loss_weights).float()
        elif params["loss"]["name"] == LossFN.FOCAL_LOSS:
            params["weights"] = torch.ones(params["n_classes"])

        # Imprimindo os pesos das classes para a loss
        print(params["weights"])
    except Exception as e:
        print(e)
        raise e


def update_models_comparison(params, trainer, training_time_hours, test_time_hours):
    """Atualiza um arquivo TXT central com as métricas de todos os modelos"""

    # Arquivo central no output root
    comparison_file = os.path.join(params["results_folder"], "models_comparison.txt")

    # Coletar as métricas finais
    best_train_acc = max(trainer.epoch_acc) if trainer.epoch_acc else 0
    best_val_acc = max(trainer.epoch_val_acc) if trainer.epoch_val_acc else 0
    best_train_f1 = max(trainer.epoch_f1) if trainer.epoch_f1 else 0
    best_val_f1 = max(trainer.epoch_val_f1) if trainer.epoch_val_f1 else 0
    best_train_iou = max(trainer.epoch_iou) if trainer.epoch_iou else 0
    best_val_iou = max(trainer.epoch_val_iou) if trainer.epoch_val_iou else 0
    best_train_mcc = max(trainer.epoch_mcc) if trainer.epoch_mcc else 0
    best_val_mcc = max(trainer.epoch_val_mcc) if trainer.epoch_val_mcc else 0

    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )

    # Verificar se o arquivo já existe para adicionar cabeçalho
    file_exists = os.path.exists(comparison_file)

    with open(comparison_file, "a", encoding="utf-8") as f:
        if not file_exists:
            # Cabeçalho para novo arquivo
            f.write("=" * 120 + "\n")
            f.write("COMPARAÇÃO DE MODELOS - MÉTRICAS DE TREINAMENTO\n")
            f.write("=" * 120 + "\n")
            f.write(
                f"{'Modelo':<20} {'Data':<12} {'Val_Acc':<8} {'Val_F1':<8} {'Val_IoU':<8} {'Val_MCC':<8} "
            )
            f.write(
                f"{'Train_Acc':<8} {'Train_F1':<8} {'Train_IoU':<8} {'Train_MCC':<8} "
            )
            f.write(f"{'Params':<12} {'Time(h)':<8}\n")
            f.write("-" * 120 + "\n")

        # Adicionar linha com métricas do modelo atual
        f.write(f"{params['model']['name']:<20} {time.strftime('%Y-%m-%d'):<12} ")
        f.write(
            f"{best_val_acc:.4f}  {best_val_f1:.4f}  {best_val_iou:.4f}  {best_val_mcc:.4f}  "
        )
        f.write(
            f"{best_train_acc:.4f}  {best_train_f1:.4f}  {best_train_iou:.4f}  {best_train_mcc:.4f}  "
        )
        f.write(f"{total_params:<12,} {training_time_hours:.2f}\n")

    print(f" Métricas adicionadas ao arquivo de comparação: {comparison_file}")

def save_final_metrics(params, trainer, training_time_hours, test_time_hours):
    """Salva as métricas finais em um arquivo TXT específico do modelo"""

    metrics_file = os.path.join(params["results_folder"], "final_metrics.txt")

    # Coletar métricas
    best_train_acc = max(trainer.epoch_acc) if trainer.epoch_acc else 0
    best_val_acc = max(trainer.epoch_val_acc) if trainer.epoch_val_acc else 0
    best_train_f1 = max(trainer.epoch_f1) if trainer.epoch_f1 else 0
    best_val_f1 = max(trainer.epoch_val_f1) if trainer.epoch_val_f1 else 0
    best_train_iou = max(trainer.epoch_iou) if trainer.epoch_iou else 0
    best_val_iou = max(trainer.epoch_val_iou) if trainer.epoch_val_iou else 0
    best_train_mcc = max(trainer.epoch_mcc) if trainer.epoch_mcc else 0
    best_val_mcc = max(trainer.epoch_val_mcc) if trainer.epoch_val_mcc else 0

    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MÉTRICAS FINAIS DO TREINAMENTO\n")
        f.write("=" * 80 + "\n")
        f.write(f"Modelo: {params['model']['name']}\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pasta: {params['results_folder']}\n")
        f.write("\n")

        f.write("CONFIGURAÇÃO:\n")
        f.write(f"  Input Size: {params['window_size']}\n")
        f.write(f"  Batch Size: {params['bs']}\n")
        f.write(f"  Learning Rate: {params['optimizer_params']['lr']}\n")
        f.write(f"  Loss: {params['loss']['name']}\n")
        f.write(f"  Optimizer: {params['optimizer_params']['optimizer']}\n")
        f.write("\n")

        f.write("MÉTRICAS DE TREINO (MELHORES):\n")
        f.write(f"  Accuracy: {best_train_acc:.4f}\n")
        f.write(f"  F1-Score: {best_train_f1:.4f}\n")
        f.write(f"  IoU: {best_train_iou:.4f}\n")
        f.write(f"  MCC: {best_train_mcc:.4f}\n")
        f.write("\n")

        f.write("MÉTRICAS DE VALIDAÇÃO (MELHORES):\n")
        f.write(f"  Accuracy: {best_val_acc:.4f}\n")
        f.write(f"  F1-Score: {best_val_f1:.4f}\n")
        f.write(f"  IoU: {best_val_iou:.4f}\n")
        f.write(f"  MCC: {best_val_mcc:.4f}\n")
        f.write("\n")

        f.write("TEMPOS DE EXECUÇÃO:\n")
        f.write(f"  Treinamento: {training_time_hours:.2f} horas\n")
        f.write(f"  Inferência: {test_time_hours:.2f} horas\n")
        f.write(f"  Total: {training_time_hours + test_time_hours:.2f} horas\n")
        f.write("\n")

        f.write("ESTATÍSTICAS DO MODELO:\n")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        f.write(f"  Total de Parâmetros: {total_params:,}\n")
        f.write(f"  Parâmetros Treináveis: {trainable_params:,}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f" Métricas finais salvas em: {metrics_file}")


if __name__ == "__main__":
    # Registra o tempo de início do treinamento
    start_time = time.time()

    # PATHS CORRIGIDOS - Compatível com Linux
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset_35")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # Params
    params = {
        "root_dir": DATASET_DIR,  # Diretório raiz dos dados
        "results_folder": OUTPUT_DIR,
        "cache": True,
        "window_size": (224, 224),  # Tamanho das imagens de entrada da rede
        "bs": 40,  # Batch size
        "n_classes": 8,  # Número de classes
        "classes": [
            "Urbano",
            "Vegetação Densa",
            "Sombra",
            "Vegetação Esparsa",
            "Agricultura",
            "Rocha",
            "Solo Exposto",
            "Água",
        ],  # Nome das classes
        "maximum_epochs": 999,  # Número de épocas de treinamento
        "save_epoch": 2,  # Salvar o modelo a cada n épocas para evitar perder o treinamento caso ocorra algum erro ou queda de energia
        "print_each": 100,  # Print each n iterations (apenas para acompanhar visualmente o treinamento)
        "augment": False,
        "cpu": None,  # CPU ou GPU. Se None, será usado GPU. Não vai funcionar com CPU
        "device": "cuda",  # GPU
        "precision": "full",  # Precisão dos cálculos. 'full' ou 'half'. 'full' é mais preciso, mas mais lento. 'half' é mais rápido, mas menos preciso. Default: 'full'
        "optimizer_params": {
            "optimizer": "ADAM",
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0,
            "epsilon": 1e-8,
            "momentum": 0.9,
        },
        "lrs_params": {
            "type": "Plateau",
            "lr_decay": 30,
            "milestones": [25, 35, 45],
            "gamma": 0.1,
        },
        "weights": "",  # Peso de cada classe para a loss. Será calculado automaticamente em seguida
        "loss": {
            "name": LossFN.TVERSKY,  # Escolha entre 'CROSS_ENTROPY' ou 'FOCAL_LOSS' 'DICE'
            "params": {
                "weights": "calculate",  # Escolha entre 'equal' ou 'calculate'. Se 'equal', os pesos serão iguais. Se 'calculate', os pesos serão calculados pelo arquivo `extra\weights_calculator.py`
                "alpha": 0.5,  # Somente para FOCAL_LOSS. Informe um valor float. Default: 0.5
                "gamma": 2.0,  # Somente para FOCAL_LOSS. Informe um valor float. Default: 2.0
            },
        },
        "patience": 10,
        "model": {
            "name": ModelChooser.DEEPLABV3PLUS,  # Escolha entre 'SEGNET_MODIFICADA' ou 'UNET' ou 'SEGFORMER' DEEPLABV3PLUS
        },
    }

    print("=" * 80)
    print("CONFIGURAÇÃO DO EXPERIMENTO - HIPERPARÂMETROS 1:1")
    print("=" * 80)
    print(f"Modelo: {params['model']['name']}")
    print(f"Input Size: {params['window_size']}")
    print(f"Número de Classes: {params['n_classes']}")
    print(f"Batch Size: {params['bs']}")
    print(f"Máximo de Épocas: {params['maximum_epochs']}")
    print(f"Otimizador: {params['optimizer_params']['optimizer']}")
    print(f"Learning Rate: {params['optimizer_params']['lr']}")
    print(f"Weight Decay: {params['optimizer_params']['weight_decay']}")
    print(f"Loss Function: {params['loss']['name']}")
    print(f"Loss Params: {params['loss']['params']}")
    print(f"Scheduler: {params['lrs_params']['type']}")
    print(f"Patience: {params['patience']}")
    print(f"Augment: {params['augment']}")
    print("=" * 80)

    # CORRIGIDO: Usar caminhos relativos ao BASE_DIR
    model_name = params["model"]["name"]
    optimizer_name = params["optimizer_params"]["optimizer"]
    weight_decay = params["optimizer_params"]["weight_decay"]
    loss_name = params["loss"]["name"]

    results_folder_name = f"K1x5noAug_{model_name}b45drop2_imgnet_{optimizer_name}{weight_decay}WD_{loss_name}1.0-0.5_noWeight"
    params["results_folder"] = os.path.join(OUTPUT_DIR, results_folder_name)

    # Criar diretório de resultados se não existir
    os.makedirs(params["results_folder"], exist_ok=True)

    print(f"Resultados serão salvos em:")
    print(f"   {os.path.abspath(params['results_folder'])}")
    print("=" * 80)

    # CORRIGIDO: Usar caminhos relativos
    image_dir = os.path.join(params["root_dir"], "images")
    label_dir = os.path.join(params["root_dir"], "labels")
    edges_dir = os.path.join(params["root_dir"], "edges")

    # CORRIGIDO: Load image and label files from .txt usando caminhos relativos
    # TRAIN FOLDS (1, 2, 3)
    train_images1 = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold1_images.txt"),
        header=None,
    ).values
    train_images2 = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold2_images.txt"),
        header=None,
    ).values
    train_images3 = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold3_images.txt"),
        header=None,
    ).values
    train_images = [
        os.path.join(image_dir, f[0])
        for f in np.concatenate([train_images1, train_images2, train_images3])
    ]

    train_labels1 = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold1_labels.txt"),
        header=None,
    ).values
    train_labels2 = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold2_labels.txt"),
        header=None,
    ).values
    train_labels3 = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold3_labels.txt"),
        header=None,
    ).values
    train_labels = [
        os.path.join(label_dir, f[0])
        for f in np.concatenate([train_labels1, train_labels2, train_labels3])
    ]

    # VALIDATION FOLD (4)
    val_images = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold4_images.txt"),
        header=None,
    ).values
    val_images = [os.path.join(image_dir, f[0]) for f in val_images]

    val_labels = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold4_labels.txt"),
        header=None,
    ).values
    val_labels = [os.path.join(label_dir, f[0]) for f in val_labels]

    # TEST FOLD (5)
    test_images = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold5_images.txt"),
        header=None,
    ).values
    test_images = [os.path.join(image_dir, f[0]) for f in test_images]

    test_labels = pd.read_table(
        os.path.join(DATASET_DIR, "folds", "fold5_labels.txt"),
        header=None,
    ).values
    test_labels = [os.path.join(label_dir, f[0]) for f in test_labels]

    print(f"Dataset carregado:")
    print(f"   Train: {len(train_images)} imagens")
    print(f"   Val: {len(val_images)} imagens")
    print(f"   Test: {len(test_images)} imagens")
    print("=" * 80)

    # Carregar os pesos de cada classe, calculados pelo arquivo `extra\weights_calculator.py`
    weights_calculator_loss(params, train_labels)

    # Create train and test sets
    train_dataset = DatasetIcmbio(
        train_images,
        train_labels,
        None,
        window_size=params["window_size"],
        cache=params["cache"],
        augmentation=params["augment"],
    )
    val_dataset = DatasetIcmbio(
        val_images,
        val_labels,
        window_size=params["window_size"],
        cache=params["cache"],
        augmentation=False,
    )
    test_dataset = DatasetIcmbio(
        test_images,
        test_labels,
        window_size=params["window_size"],
        cache=params["cache"],
        augmentation=False,
    )

    # Load dataset classes in pytorch dataloader handler object
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params["bs"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params["bs"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params["bs"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(model_name=params["model"]["name"], params=params)

    print("CONFIGURAÇÃO DA ARQUITETURA:")
    print(f"   Model: {params['model']['name']}")
    print(f"   Encoder: EfficientNet-B4 (ImageNet)")
    print(f"   Device: {params['device']}")
    print(f"   Precision: {params['precision']}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print("=" * 80)

    loader = {
        "train": train_loader,
        "test": test_loader,
        "val": val_loader,
    }

    # CORRIGIDO: Checkpoint path (comentado, iniciando do zero)
    # Se quiser carregar um checkpoint anterior, descomente e ajuste o path:
    # cbkp = os.path.join(OUTPUT_DIR, "K1x10augRot90_unetb05noDrop_imgnet_ADAM0WD_focal_loss", "best_epoch7573.pth.tar")
    cbkp = None

    trainer = Trainer(model, loader, params, cbkp=cbkp)

    patCB = Callback(patience=params["patience"], min_value=60)

    # Start the training.
    print("INICIANDO TREINAMENTO...")
    print("=" * 80)

    for epoch in range(trainer.last_epoch + 1, params["maximum_epochs"]):
        acc_train, f1score_train, mcc_train, iou_train = trainer.train()
        acc_val, f1score_val, mcc_val, iou_val = trainer.validate(stride=64)

        trainer.epoch_acc.append(acc_train)
        trainer.epoch_val_acc.append(acc_val)
        trainer.epoch_f1.append(f1score_train)
        trainer.epoch_val_f1.append(f1score_val)
        trainer.epoch_mcc.append(mcc_train)
        trainer.epoch_val_mcc.append(mcc_val)
        trainer.epoch_iou.append(iou_train)
        trainer.epoch_val_iou.append(iou_val)

        trainer.plot_metrics(params["results_folder"])

        if trainer.scheduler is not None:
            trainer.scheduler.step(iou_val)

        if patCB.patience_iou_val(iou_val):
            trainer.save(os.path.join(params["results_folder"], "best_epoch.pth.tar"))

        if patCB.COUNTER == patCB.PATIENCE:
            print(f"PATIENCE :: Training Terminated | Best Epoch = {epoch-10}")
            break

    np.savez(
        os.path.join(params["results_folder"], "metrics_train.npz"),
        acc_train=trainer.epoch_acc,
        acc_val=trainer.epoch_val_acc,
        f1score_train=trainer.epoch_f1,
        f1score_val=trainer.epoch_val_f1,
        iou_train=trainer.epoch_iou,
        iou_val=trainer.epoch_val_iou,
    )

    # Registra o tempo de término do treinamento
    end_time = time.time()
    # Calcula o tempo gasto em horas
    training_time = end_time - start_time
    training_time_hours = training_time / 3600.0
    print("=" * 80)
    print(f"Tempo gasto treinando: {training_time_hours:.2f} horas")
    print("=" * 80)

    # Carregar melhor modelo para teste final
    print("INICIANDO TESTE FINAL COM MELHOR MODELO...")
    trainer = Trainer(
        model,
        loader,
        params,
        cbkp=os.path.join(params["results_folder"], "best_epoch.pth.tar"),
    )

    all_preds = trainer.test(stride=64, all=True)

    print("=" * 80)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
