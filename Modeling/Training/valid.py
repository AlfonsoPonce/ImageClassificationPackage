from Metrics.ClassificationMetrics import ClassificationMetrics
import torch

def validation(device, model, valid_loader, metric_name: str, criterion):
    model.eval()
    metricas = ClassificationMetrics()
    
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    with torch.no_grad():
        for input, labels in valid_loader:
            input = input.to(device) 
            outputs = model(input) 
            
            predictions = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels.to(device))
            
            all_predictions.extend(list(predictions.cpu().numpy()))
            all_targets.extend(list(labels.cpu().numpy()))
            
            running_loss += loss.item()
            
    all_predictions = torch.Tensor(all_predictions)
    all_targets = torch.Tensor(all_targets)
    
    
    metric_function = metricas.__getattribute__(metric_name)
    
    return running_loss / len(valid_loader), metric_function(all_predictions, all_targets).item()