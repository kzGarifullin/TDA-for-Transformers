import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score,  matthews_corrcoef, accuracy_score
from tqdm import tqdm

def baseline_epoch_validation(model, classifier, val_loader, device):
    classifier.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels  in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[1]
            logits = classifier(outputs)
            val_preds.extend(torch.argmax(logits, dim=1).tolist())
            val_labels.extend(labels.tolist())
    val_acc = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    mcc = matthews_corrcoef(val_labels, val_preds)
    return val_acc, precision, recall, mcc


# Baseline approach
def train_baseline(model, classifier, criterion, optimizer, train_loader, val_loader, num_epochs, device):
    # Freeze the parameters of the BERT model
    for param in model.parameters():
        param.requires_grad = False

    classifier.train()
    for epoch in range(num_epochs):

        avg_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[1]  # Get pooled output from BERT
            logits = classifier(outputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss

        val_acc, precision, recall, mcc = baseline_epoch_validation(model, classifier, val_loader, device) 
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc}")
        print("Validation MCC:", mcc)
        print("Train loss: ", avg_loss.item())
        print("_______________________")


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.linear(x)
