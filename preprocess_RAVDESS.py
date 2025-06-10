import torch
import torchaudio
import os
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np

def load_audio(file_path):
    """Load and preprocess audio file."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy()
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        return None

def get_all_wav_files(dataset_path):
    """Recursively get all WAV files from all actor directories."""
    wav_files = []
    for actor_dir in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_dir)
        print(f"Checking directory: {actor_path}")  # Debugging print statement
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    full_path = os.path.join(actor_path, file)
                    wav_files.append(full_path)
                    print(f"Found WAV file: {full_path}")  # Debugging print statement
    return wav_files

def extract_embeddings():
    # Initialize model and feature extractor
    model = Wav2Vec2Model.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        ignore_mismatched_sizes=True,
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Path to RAVDESS dataset
    dataset_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS"
    
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Dataset path: {dataset_path}")  # Debugging print statement

    # Get list of all WAV files from all actor directories
    wav_files = get_all_wav_files(dataset_path)
    print(f"Found {len(wav_files)} WAV files in dataset")
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {dataset_path}")

    embeddings = []
    labels = []
    processed_files = []

    for file_path in tqdm(wav_files, desc="Processing audio files"):
        try:
            # Extract emotion label from filename
            filename = os.path.basename(file_path)
            parts = filename.split("-")
            if len(parts) < 3:
                print(f"Skipping {filename}: Invalid filename format")
                continue
                
            emotion = int(parts[2]) - 1  # Convert to 0-based index
            
            # Load and preprocess audio
            waveform = load_audio(file_path)
            if waveform is None:
                continue
                
            # Prepare input for model
            inputs = feature_extractor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings.append(embedding.squeeze())
            labels.append(emotion)
            processed_files.append(file_path)
            
            # Print shape of current embedding for debugging
            if len(embeddings) == 1:
                print(f"Shape of first embedding: {embedding.squeeze().shape}")
                
            # Add this inside the processing loop after getting the embedding
            if len(embeddings) <= 2:  # Check first two embeddings
                print(f"File: {filename}")
                print(f"Embedding stats:")
                print(f"- Shape: {embedding.squeeze().shape}")
                print(f"- Mean: {embedding.mean():.4f}")
                print(f"- Std: {embedding.std():.4f}")
                print(f"- Min: {embedding.min():.4f}")
                print(f"- Max: {embedding.max():.4f}")
                print("---")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    if len(embeddings) == 0:
        raise ValueError("No embeddings were successfully extracted!")

    # Convert lists to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Save embeddings and labels
    save_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    # Check if files already exist and delete them
    embeddings_file = os.path.join(save_path, 'embeddings.npy')
    labels_file = os.path.join(save_path, 'labels.npy')
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
    if os.path.exists(labels_file):
        os.remove(labels_file)
    
    # Save new files with specified names
    np.save(os.path.join(save_path, 'embeddings_RAVDESS.npy'), embeddings)
    np.save(os.path.join(save_path, 'labels_RAVDESS.npy'), labels)
    
    # Save processed files list for reference
    with open(os.path.join(save_path, 'processed_files.txt'), 'w') as f:
        for file_path in processed_files:
            f.write(f"{file_path}\n")
    
    print(f"Successfully processed {len(embeddings)} files")
    print(f"Saved embeddings shape: {embeddings.shape}")
    print(f"Saved labels shape: {labels.shape}")
    print(f"Files saved to: {save_path}")
    
    return embeddings, labels

if __name__ == "__main__":
    try:
        extract_embeddings()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

# Verify files are saved
save_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition"
print("Files in directory:")
for file in os.listdir(save_path):
    if file.startswith('embeddings_RAVDESS') or file.startswith('labels_RAVDESS'):
        print(f"- {file}")
