import numpy as np
import torch
import torch.nn as nn
from train import FeedForwardNN

# Paths
moviescope_embeddings_path = r"D:\Moviescope\embeddings_MOVIESCOPE.npy"  # <-- updated path
classifier_weights_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\best_classifier_model_RAVDESS.pth"
save_softmax_path = r"D:\Moviescope\moviescope_classifier_softmax.npy"
save_penultimate_path = r"D:\Moviescope\moviescope_classifier_penultimate.npy"

# Model parameters (must match your classifier)
input_dim = 1024
hidden_dim = 1024
output_dim = 8

def extract_classifier_outputs():
    # Load embeddings (shape: [num_files, num_segments, 1024])
    embeddings = np.load(moviescope_embeddings_path, allow_pickle=True)

    # Load classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(classifier_weights_path, map_location=device))
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1)

    all_classifier_penultimate = []
    all_classifier_softmax = []

    with torch.no_grad():
        for file_embs in embeddings:
            file_softmax = []
            file_penultimate = []
            for seg_emb in file_embs:
                x = torch.FloatTensor(seg_emb).unsqueeze(0).to(device)  # [1, 1024]
                # Forward pass up to penultimate layer
                x1 = model.relu(model.bn1(model.fc1(x)))
                x1 = model.dropout(x1)
                x2 = model.relu(model.bn2(model.fc2(x1)))
                x2 = model.dropout(x2)
                penultimate = x2.squeeze(0).cpu().numpy()  # [hidden_dim//2]
                logits = model.fc3(x2)
                probs = softmax(logits).squeeze(0).cpu().numpy()  # [output_dim]
                file_penultimate.append(penultimate)
                file_softmax.append(probs)
            all_classifier_penultimate.append(np.stack(file_penultimate))
            all_classifier_softmax.append(np.stack(file_softmax))

    np.save(save_penultimate_path, np.array(all_classifier_penultimate, dtype=object))
    np.save(save_softmax_path, np.array(all_classifier_softmax, dtype=object))

    print(f"Saved classifier softmax outputs to {save_softmax_path}")
    print(f"Saved classifier penultimate embeddings to {save_penultimate_path}")

if __name__ == "__main__":
    extract_classifier_outputs()