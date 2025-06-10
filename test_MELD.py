import torch
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

# Custom EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        try:
            torch.save(model.state_dict(), self.path)
        except Exception as e:
            self.trace_func(f"Error saving checkpoint: {str(e)}")
        self.val_loss_min = val_loss

# Custom dataset class for handling embeddings
class EmotionDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Define the fully connected feed-forward neural network
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.4):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_and_evaluate(model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, device, num_epochs=35):
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_losses = []
    val_losses = []
    test_losses = []

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Apply weight clipping
            for param in model.parameters():
                param.data.clamp_(-0.5, 0.5)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {train_accuracy * 100:.2f}%")

        # Evaluate on the validation set
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Evaluate on the test set
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss / len(test_loader))
        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

        # Step the scheduler
        scheduler.step()

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_accuracies, val_accuracies, test_accuracies, train_losses, val_losses, test_losses

def main():
    # Load the pre-extracted embeddings and labels for train, validation, and test sets
    print("Loading embeddings and labels...")
    train_embeddings_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/embeddings_train_MELD.npy'
    train_labels_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/labels_train_MELD.npy'
    val_embeddings_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/embeddings_dev_MELD.npy'
    val_labels_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/labels_dev_MELD.npy'
    test_embeddings_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/embeddings_test_MELD.npy'
    test_labels_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/labels_test_MELD.npy'

    train_embeddings = np.load(train_embeddings_path)
    train_labels = np.load(train_labels_path)
    val_embeddings = np.load(val_embeddings_path)
    val_labels = np.load(val_labels_path)
    test_embeddings = np.load(test_embeddings_path)
    test_labels = np.load(test_labels_path)
    
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation embeddings shape: {val_embeddings.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Apply StandardScaler
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)
    test_embeddings = scaler.transform(test_embeddings)
    
    print("Standard scaling applied to embeddings")
    print(f"Mean after scaling: {train_embeddings.mean():.6f}")
    print(f"Std after scaling: {train_embeddings.std():.6f}")

    # Create data loaders
    train_dataset = EmotionDataset(train_embeddings, train_labels)
    val_dataset = EmotionDataset(val_embeddings, val_labels)
    test_dataset = EmotionDataset(test_embeddings, test_labels)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    input_dim = train_embeddings.shape[1]  # Updated input dimension
    hidden_dim = 1024
    output_dim = 7  # Number of emotion classes in MELD

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_embeddings)):
        print(f"Fold {fold + 1}")

        # Create data loaders for the current fold
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = FeedForwardNN(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        train_accuracies, val_accuracies, test_accuracies, train_losses, val_losses, test_losses = train_and_evaluate(
            model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, device, num_epochs=35
        )

        fold_results.append({
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_accuracies': test_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses
        })

    # Find the minimum length across all folds
    min_length = min(
        min(len(result['train_accuracies']) for result in fold_results),
        min(len(result['val_accuracies']) for result in fold_results),
        min(len(result['test_accuracies']) for result in fold_results)
    )

    # Truncate all results to the minimum length
    for result in fold_results:
        result['train_accuracies'] = result['train_accuracies'][:min_length]
        result['val_accuracies'] = result['val_accuracies'][:min_length]
        result['test_accuracies'] = result['test_accuracies'][:min_length]
        result['train_losses'] = result['train_losses'][:min_length]
        result['val_losses'] = result['val_losses'][:min_length]
        result['test_losses'] = result['test_losses'][:min_length]

    # Calculate averages with the truncated arrays
    avg_train_accuracies = np.mean([result['train_accuracies'] for result in fold_results], axis=0)
    avg_val_accuracies = np.mean([result['val_accuracies'] for result in fold_results], axis=0)
    avg_test_accuracies = np.mean([result['test_accuracies'] for result in fold_results], axis=0)
    avg_train_losses = np.mean([result['train_losses'] for result in fold_results], axis=0)
    avg_val_losses = np.mean([result['val_losses'] for result in fold_results], axis=0)
    avg_test_losses = np.mean([result['test_losses'] for result in fold_results], axis=0)

    # Plot average training vs validation accuracy
    plt.figure()
    plt.plot(range(1, len(avg_train_accuracies) + 1), avg_train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(avg_val_accuracies) + 1), avg_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Average Training vs Validation Accuracy_MELD')
    plt.legend()
    plt.savefig('avg_train_vs_val_accuracy_MELD.png')
    plt.close()

    # Plot average training vs test accuracy
    plt.figure()
    plt.plot(range(1, len(avg_train_accuracies) + 1), avg_train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(avg_test_accuracies) + 1), avg_test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Average Training vs Test Accuracy_MELD')
    plt.legend()
    plt.savefig('avg_train_vs_test_accuracy_MELD.png')
    plt.close()

    # Plot average train vs average test loss
    plt.figure()
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Train Loss')
    plt.plot(range(1, len(avg_test_losses) + 1), avg_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Train vs Test Loss_MELD')
    plt.legend()
    plt.savefig('avg_train_vs_test_loss_MELD.png')
    plt.close()

    # Print final test accuracy and test loss
    final_test_accuracy = avg_test_accuracies[-1]
    final_test_loss = avg_test_losses[-1]
    print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
    print(f"Final Test Loss: {final_test_loss:.4f}")

if __name__ == "__main__":
    main()