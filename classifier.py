import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dataloaderSCAPIS import EmbeddingDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DominanceClassifier(nn.Module):
    def __init__(self, in_dim=320, hidden_dim1=128, hidden_dim2=64, num_classes=2, dropout_p=0.3):
        super().__init__()
        
        # Hidden layer
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        """
        Forward pass through classifier head.
        Returns both logits and penultimate embeddings."""

        hidden = self.fc1(x)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.fc2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.dropout2(hidden)

        logits = self.fc3(hidden)

        return logits, hidden  # hidden = penultimate embedding
        


#train classifier
if __name__ == "__main__":
    # Example usage
    nr_epochs = 500
    fold = 0
    results_dir = "../results"

    model = DominanceClassifier(in_dim=320, hidden_dim1=128, hidden_dim2=64, num_classes=2, dropout_p=0.3)
    dataset = EmbeddingDataset(f"../dataSC-embeddings/embeddings_{fold}_test.npy", f"../dataSC-embeddings/labels_{fold}_test.npy")
    datasetVal = EmbeddingDataset(f"../dataSC-embeddings/embeddings_{fold}_val.npy", f"../dataSC-embeddings/labels_{fold}_val.npy")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    dataloaderVal = DataLoader(datasetVal, batch_size=8, shuffle=False)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    #save results for plotting
    train_losses = []
    val_losses = []

    model.train()
    print("podatkov: ", len(dataloader))
    for epoch in range(nr_epochs):  # number of epochs
        total_loss = 0
        model.train()
        for batch_embeddings, batch_labels in dataloader:
            #optimizer.zero_grad()
            logits, _ = model(batch_embeddings)
          #  print("logits shape:", logits.shape)
          #  print("batch_labels shape:", batch_labels.squeeze().shape)
            l = loss(logits, batch_labels.squeeze())
            l.backward()
            optimizer.step()
            total_loss += l.item()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{nr_epochs}, train loss: {avg_loss:.4f}")

        # Validate the model
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_embeddings, batch_labels in dataloaderVal:
                logits, _ = model(batch_embeddings)
                l = loss(logits, batch_labels.squeeze())
                total_val_loss += l.item()
            avg_val_loss = total_val_loss / len(dataloaderVal)
            val_losses.append(avg_val_loss)
            #print(f".    val loss: {avg_val_loss:.4f}")
    
    # Save the metrics
    pd.DataFrame({"epoch": np.arange(1, nr_epochs + 1),
                  "train_loss": train_losses, 
                  "val_loss": val_losses}).to_csv(f"{results_dir}/metrics_{fold}.csv")
    
    # Save the trained model
    torch.save(model, f"{results_dir}/dominance_classif_{fold}.pth")

    # Plot training and validation loss curves
    fig = plt.figure()
    plt.plot(np.arange(1, nr_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(np.arange(1, nr_epochs + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(f"{results_dir}/train_curves_{fold}.png")