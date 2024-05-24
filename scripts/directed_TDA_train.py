from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn

def directed_epoch_validation(model, classifier, val_loader, feat_val_loader, device):
    classifier.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():

        for data1, data2 in tqdm(zip(val_loader, feat_val_loader)):
            inputs, labels = data1
            feat, labels2 = data2
            inputs, labels = inputs.to(device), labels.to(device)
            feat = feat.to(device)
            outputs = model(inputs)[1]
            concat_tensor = torch.cat((outputs, feat), dim=1)
            logits = classifier(concat_tensor)
            val_preds.extend(torch.argmax(logits, dim=1).tolist())
            val_labels.extend(labels.tolist())
    val_acc = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    mcc = matthews_corrcoef(val_labels, val_preds)

    return val_acc, precision, recall, mcc

from tqdm import tqdm

class Directed_classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Directed_classifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.linear(x)


def train_directed(model, optimizer, criterion, classifier, train_loader, val_loader, feat_loader, feat_val_loader, num_epochs, device):
    for param in model.parameters():
        param.requires_grad = False

    classifier.train()
    for epoch in range(num_epochs):
        avg_loss = 0
        for data1, data2 in tqdm(zip(train_loader, feat_loader)):
            inputs, labels = data1 
            feat, labels2 = data2
            inputs, labels = inputs.to(device), labels.to(device)
            feat = feat.to(device)
            outputs = model(inputs)[1]  # Get pooled output from BERT
            concat_tensor = torch.cat((outputs, feat), dim=1)
            logits = classifier(concat_tensor)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss

        val_acc, precision, recall, mcc = directed_epoch_validation(model, classifier, val_loader, feat_val_loader,device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc}")
        print("Validation MCC:", mcc)
        print("Train loss: ", avg_loss.item())
        print("_______________________")
