from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


class Plots:
    def __init__(self, output_dir:Path) -> None:
        self.output_dir = output_dir
    
    def confusion_matrix_plot(self, cm, class_list: list):
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
        plt.title('Confusion Matrix')
        disp.plot()
        
        plt.savefig(self.output_dir.joinpath('Best_Weights_Confusion_Matrix.jpg'))
        
    def loss_plot(self, train_loss_list_over_epochs: list, val_loss_list_over_epochs: list):
        plt.figure()
        plt.plot(range(len(train_loss_list_over_epochs)), train_loss_list_over_epochs)
        plt.plot(range(len(val_loss_list_over_epochs)), val_loss_list_over_epochs)
        plt.title('Loss Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim(0)  
        plt.ylim(0)
        plt.legend(['Train Loss', 'Val Loss'])
        plt.savefig(self.output_dir.joinpath('Loss_Plot.jpg'))
    
    def metric_plot(self, metric_name: str, train_metric_list_over_epochs: list, val_metric_list_over_epochs: list):
        plt.figure()
        plt.plot(range(len(train_metric_list_over_epochs)), train_metric_list_over_epochs)
        plt.plot(range(len(val_metric_list_over_epochs)), val_metric_list_over_epochs)
        plt.title(f'{metric_name} Plot')
        plt.xlabel('Epochs')
        plt.ylabel(f'{metric_name}')
        plt.xlim(0)  
        plt.ylim(0)
        plt.legend([f'Train {metric_name}', f'Val {metric_name}'])
        plt.savefig(self.output_dir.joinpath(f'{metric_name}_Plot.jpg'))
        
        