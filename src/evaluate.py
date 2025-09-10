import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model,loader,threshold=0.5):
    model.eval()
    all_pred, all_labels = [],[]
    with torch.no_grad():
        for X,y in loader:
            y_pred = model(X)
            predicted = (y_pred>=threshold).float()
            all_pred.extend(predicted.numpy())
            all_labels.extend(y.numpy())
            
    acc = accuracy_score(all_labels,all_pred)
    prec = precision_score(all_labels,all_pred,zero_division=0)
    rec = recall_score(all_labels,all_pred,zero_division=0)
    f1 = f1_score(all_labels,all_pred,zero_division=0)
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return acc, prec, rec, f1