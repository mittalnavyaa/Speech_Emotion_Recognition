from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate  # For pretty printing results
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def add_gaussian_noise(data, noise_factor=0.1):
    noise = np.random.randn(*data.shape) * noise_factor
    return data + noise

def add_frequency_masking(data, mask_factor=0.1):
    num_features = data.shape[1]
    mask_size = int(num_features * mask_factor)
    mask_start = np.random.randint(0, num_features - mask_size)
    data[:, mask_start:mask_start + mask_size] = 0
    return data

def add_time_masking(data, mask_factor=0.1):
    num_samples = data.shape[0]
    mask_size = int(num_samples * mask_factor)
    mask_start = np.random.randint(0, num_samples - mask_size)
    data[mask_start:mask_start + mask_size, :] = 0
    return data

def add_small_shifts(data, shift_factor=0.1):
    shift = np.random.randint(-int(data.shape[1] * shift_factor), int(data.shape[1] * shift_factor))
    return np.roll(data, shift, axis=1)

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

        # Apply noise augmentations after each epoch
        for inputs, targets in train_loader:
            inputs = add_gaussian_noise(inputs.cpu().numpy(), noise_factor=0.1)
            inputs = add_frequency_masking(inputs, mask_factor=0.1)
            inputs = add_time_masking(inputs, mask_factor=0.1)
            inputs = add_small_shifts(inputs, shift_factor=0.1)
            inputs = torch.FloatTensor(inputs).to(device)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_accuracies, val_accuracies, test_accuracies, train_losses, val_losses, test_losses

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', value=0.):
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)

    padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i, :maxlen] = seq[:maxlen]
        else:
            padded_sequences[i, :len(seq)] = seq
    return padded_sequences

def main():
    # Load the pre-extracted embeddings
    print("Loading embeddings and labels...")
    embeddings_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/embeddings_RAVDESS.npy'
    labels_path = 'C:/Users/navya/wav2vec2-lg-xlsr-en-speech-emotion-recognition/labels_RAVDESS.npy'
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Apply StandardScaler
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    print("Standard scaling applied to embeddings")
    print(f"Mean after scaling: {embeddings.mean():.6f}")
    print(f"Std after scaling: {embeddings.std():.6f}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Further split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set size after split: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Cross-validation setup
    k_folds = 5
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Lists to store cross-validation results
    cv_train_accuracies = []
    cv_val_accuracies = []
    cv_test_accuracies = []
    cv_train_losses = []
    cv_val_losses = []
    cv_test_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"Fold {fold+1}/{k_folds}")

        # Create data loaders for the current fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        train_dataset = EmotionDataset(X_train_fold, y_train_fold)
        val_dataset = EmotionDataset(X_val_fold, y_val_fold)
        test_dataset = EmotionDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize the model, loss function, and optimizer
        input_dim = 1024  # Updated input dimension
        hidden_dim = 1024
        output_dim = 8
        model = FeedForwardNN(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        # Adam optimizer with L2 regularization (weight decay)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        # Train and evaluate the model
        train_accuracies, val_accuracies, test_accuracies, train_losses, val_losses, test_losses = train_and_evaluate(
            model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, device, num_epochs=35
        )

        cv_train_accuracies.append(train_accuracies)
        cv_val_accuracies.append(val_accuracies)
        cv_test_accuracies.append(test_accuracies)
        cv_train_losses.append(train_losses)
        cv_val_losses.append(val_losses)
        cv_test_losses.append(test_losses)

    # Pad sequences to ensure they have the same length
    cv_train_accuracies = pad_sequences(cv_train_accuracies)
    cv_val_accuracies = pad_sequences(cv_val_accuracies)
    cv_test_accuracies = pad_sequences(cv_test_accuracies)
    cv_train_losses = pad_sequences(cv_train_losses)
    cv_val_losses = pad_sequences(cv_val_losses)
    cv_test_losses = pad_sequences(cv_test_losses)

    # Calculate the mean for each epoch across all folds
    avg_train_accuracies = np.mean(cv_train_accuracies, axis=0)
    avg_val_accuracies = np.mean(cv_val_accuracies, axis=0)
    avg_test_accuracies = np.mean(cv_test_accuracies, axis=0)
    avg_train_losses = np.mean(cv_train_losses, axis=0)
    avg_val_losses = np.mean(cv_val_losses, axis=0)
    avg_test_losses = np.mean(cv_test_losses, axis=0)

    # Final evaluation on the test set
    test_dataset = EmotionDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"Final Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

    # Additional evaluation metrics
    print("Classification Report:")
    print(classification_report(all_targets, all_predictions))

    # Generate and save confusion matrix
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    cm = confusion_matrix(all_targets, all_predictions)
    epsilon = 1e-10  # Small value to avoid division by zero
    cm_percentage = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + epsilon) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Plot average training vs validation accuracy across all folds
    plt.figure()
    plt.plot(range(1, len(avg_train_accuracies) + 1), avg_train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(avg_val_accuracies) + 1), avg_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Average Training vs Validation Accuracy')
    plt.legend()
    plt.savefig('avg_train_vs_val_accuracy.png')
    plt.close()

    # Plot average training vs test accuracy across all folds
    plt.figure()
    plt.plot(range(1, len(avg_train_accuracies) + 1), avg_train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(avg_test_accuracies) + 1), avg_test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Average Training vs Test Accuracy')
    plt.legend()
    plt.savefig('avg_train_vs_test_accuracy.png')
    plt.close()

    # Plot average train vs average test loss across all folds
    plt.figure()
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Train Loss')
    plt.plot(range(1, len(avg_test_losses) + 1), avg_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Train vs Test Loss')
    plt.legend()
    plt.savefig('avg_train_vs_test_loss.png')
    plt.close()

if __name__ == "__main__":
    main()