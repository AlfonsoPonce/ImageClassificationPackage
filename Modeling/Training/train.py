'''
Module that implements model training

Author: Alfonso Ponce Navarro
Date: 08/02/2024
'''
from pathlib import Path
import torch
from .valid import validation
from Metrics.ClassificationMetrics import ClassificationMetrics
from .plots import Plots
import time
import copy
# if __name__ == '__main__':
def train(
        model,
        class_list: list,
        train_config_dict: dict,
        train_loader,
        valid_loader,
        output_dir: Path) -> tuple:
    '''
    Function that implements model training.

    :param model: torch detection model
    :param class_list: list of class labels
    :param train_config_dict: configuration training dictionary
    :param train_loader: torch train dataloader
    :param valid_loader: torch valid dataloader
    :param output_dir: Path where trained model will be stored
    :return: Best and last metrics.
    '''

    assert isinstance(train_config_dict['optimizer'], torch.optim.Optimizer)
    assert isinstance(train_config_dict['epochs'], int)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(valid_loader.dataset)}\n")
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    
    metric = ClassificationMetrics()
    optimizer = train_config_dict['optimizer']
    num_epochs = train_config_dict['epochs']
    criterion = train_config_dict['criterion']
    if train_config_dict['scheduler'] != 'None':
        scheduler = train_config_dict['scheduler']
    
    best_val_metric = 0.0
    train_loss_list_over_epochs = []
    val_loss_list_over_epochs = []
    train_metric_list_over_epochs = []
    val_metric_list_over_epochs = []
    plotter = Plots(output_dir)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(list(predictions.cpu().numpy()))
            all_targets.extend(list(labels.cpu().numpy()))
            
            
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
                
        if scheduler != 'None':
            scheduler.step()
        
        
        all_predictions = torch.Tensor(all_predictions)
        all_targets = torch.Tensor(all_targets)
       
    
        metric_function = metric.__getattribute__(train_config_dict['metric'])
    
        train_metric = metric_function(all_predictions, all_targets).item()
        val_loss, curr_val_metric = validation(device, model, valid_loader, train_config_dict['metric'], criterion)
        
        train_loss_list_over_epochs.append(running_loss / len(train_loader))
        val_loss_list_over_epochs.append(val_loss)
        train_metric_list_over_epochs.append(train_metric)
        val_metric_list_over_epochs.append(curr_val_metric)
        
        print(f'=====================================Epoch {epoch + 1}=====================================')
        print(f'Train_loss: {running_loss / len(train_loader):.3f} - Train_{train_config_dict["metric"]}: {train_metric:.3f}')
        print(f'Val_loss: {val_loss:.3f} - Current Valid_{train_config_dict["metric"]}: {curr_val_metric:.3f} - Best Valid_{train_config_dict["metric"]}:{best_val_metric:.3f}')
        print(f'Execution time: {time.time()-start_time:.3f} segs.')
        print(f'=================================================================================')
        
        if curr_val_metric > best_val_metric:
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, output_dir.joinpath('best_weights.pth'))
            
            plotter.confusion_matrix_plot(metric.confusion_matrix(all_predictions, all_targets), class_list)
            
            best_val_metric = curr_val_metric
    plotter.loss_plot(train_loss_list_over_epochs, val_loss_list_over_epochs)
    plotter.metric_plot(train_config_dict['metric'], train_metric_list_over_epochs, val_metric_list_over_epochs)   
    print('Finished Training')
    

