import sys
sys.path.append('D:\\Projetos\\icmbio\\')
import matplotlib
matplotlib._log.disabled = True

import pandas as pd
import logging, os, io
from pathlib import Path
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from trainer import Trainer
from utils import convert_to_color
import torch
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

RESULTS_PATH = 'D:\\Projetos\\icmbio\\results'

logging.basicConfig(filename=f'{RESULTS_PATH}/log.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Turn off sina logging
for name in ["matplotlib", "matplotlib.font", "matplotlib.pyplot"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True

class TestTrained:

    def __init__(self, test_result_path: str, classes, name: str):
        self.test = np.load(test_result_path, allow_pickle=True)
        self.classes = classes
        self.name = name
        Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        
        logging.info('\n\n--- Iniciando o teste para gerar relatórios ---')
        logging.info(f'--- {test_result_path} ---')

        print(f'--- Carregando o arquivo com as previsões e ground truths ---')
        self.y_pred = self.test.f.arr_0.item().get('all_preds')
        self.y_true = self.test.f.arr_0.item().get('all_gts')

        self.y_pred_ = np.concatenate([p.ravel() for p in self.y_pred])
        self.y_true_ = np.concatenate([p.ravel() for p in self.y_true])

        # self.imprimir_array_em_arquivo(f'{RESULTS_PATH}/true_pred.csv', self.y_true_[:100000], self.y_pred_[:100000])
        
        self.acc = self.test.f.arr_0.item().get('acc')
        logging.info(f'--- Accuracy: {self.acc} ---')

    # def imprimir_array_em_arquivo(self, nome_arquivo, y_true, y_pred):
    #     df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    #     df.to_csv(nome_arquivo, index=False, header=False)

    def run(self):
        print(f'--- Gerando a Matriz de Confusão ---')
        logging.info(f'--- Geranco a Matriz de Confusão ---')
        cm = confusion_matrix(self.y_true_, self.y_pred_, labels=range(len(self.classes)))
        
        fig, ax = plot_confusion_matrix(
            conf_mat=cm, 
            figsize=(8,8), 
            cmap=plt.cm.Blues,
            colorbar=True,
            show_absolute=False,
            show_normed=True,
        )

        if self.classes is not None:
            tick_marks = np.arange(len(self.classes))
            plt.xticks(tick_marks, self.classes, rotation=45)
            plt.yticks(tick_marks, self.classes)

        # plt.xlabel('Predictions', fontsize=12)
        # plt.ylabel('Target', fontsize=12)
        # plt.title('Confusion Matrix', fontsize=12)
        # plt.show()
        logging.info(f'--- Exportando Matriz de Confusão ---')
        fig.savefig(f"{RESULTS_PATH}/cm_{self.name}", dpi=fig.dpi, bbox_inches='tight')
        plt.close(fig)

        logging.info(f'--- Convertendo Array para Tensor ---')
        y_true_ = torch.from_numpy(self.y_true_)
        y_pred_ = torch.from_numpy(self.y_pred_)

        accuracy = Accuracy(task="multiclass", num_classes=len(self.classes))
        # DOC: IOU = true_positive / (true_positive + false_positive + false_negative)
        jaccard = MulticlassJaccardIndex(task="multiclass", num_classes=len(self.classes), average='macro')
        acc = accuracy(y_true_, y_pred_)
        jac = jaccard(y_true_, y_pred_)
        kap = cohen_kappa_score(y_true_, y_pred_)
        jac_all = metrics.jaccard_score(y_true_, y_pred_, average=None)

        logging.info(f'--- Accuracy: {acc}\t IoU: {jac}\t Kappa: {kap} ---')
        logging.info(f'--- IoU by Class: {jac_all} ---')
        
        report = metrics.classification_report(y_true_, y_pred_, target_names=self.classes)
        logging.info(f'--- Relatório ---\n{report}\n')

    # def _report(self, TN, FP, FN, TP):
    #     TPR = TP/(TP+FN) if (TP+FN)!=0 else 0
    #     TNR = TN/(TN+FP) if (TN+FP)!=0 else 0
    #     PPV = TP/(TP+FP) if (TP+FP)!=0 else 0
    #     report = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 
    #             'TPR': TPR, 'Recall': TPR, 'Sensitivity': TPR,
    #             'TNR' : TNR, 'Specificity': TNR,
    #             'FPR': FP/(FP+TN) if (FP+TN)!=0 else 0,
    #             'FNR': FN/(FN+TP) if (FN+TP)!=0 else 0,
    #             'PPV': PPV, 'Precision': PPV,
    #             'F1 Score': 2*(PPV*TPR)/(PPV+TPR)
    #             }
    #     return report

if __name__ == '__main__':
    # classes = ["Desenvolvimento", "Floresta", "Piscina", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]
    # test_result_path = 'D:\\Projetos\\icmbio\\tmp\\20230626_cross_entropy_100_epoch_no_weights_efficientnet\\segnet256_test_result.npz', #3.2

    paths = [
        {"test_result_path": '20230208_cross_entropy_100_epoch_weights_1',                "name": 'cenario_11', "classes": ["Desenvolvimento", "Floresta", "Piscina", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230209_cross_entropy_100_epoch_weights_calc',             "name": 'cenario_12', "classes": ["Desenvolvimento", "Floresta", "Piscina", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230217_cross_entropy_100_epoch_weights_calc',             "name": 'cenario_21', "classes": ["Desenvolvimento", "Floresta", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230219_cross_entropy_100_epoch_weights_calc',             "name": 'cenario_22', "classes": ["Desenvolvimento", "Floresta", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230221_cross_entropy_100_epoch_weights_efficientnet',     "name": 'cenario_31', "classes": ["Desenvolvimento", "Floresta", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230626_cross_entropy_100_epoch_no_weights_efficientnet',  "name": 'cenario_32', "classes": ["Desenvolvimento", "Floresta", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230711_cross_entropy_100_epoch_segnet_focalloss',         "name": 'cenario_41', "classes": ["Desenvolvimento", "Floresta", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
        {"test_result_path": '20230709_cross_entropy_100_epoch_unet_focalloss',           "name": 'cenario_42', "classes": ["Desenvolvimento", "Floresta", "Sombra", "Regeneração", "Agricultura", "Rocha", "Solo Exposto", "Água"]},
    ]

    # print(paths[0].split('\\')[-2])

    [TestTrained(
        "D:\\Projetos\\icmbio\\tmp\\{}\\segnet256_test_result.npz".format(x['test_result_path']), 
        x['classes'], 
        x['name']
    ).run() 
    for x in paths]