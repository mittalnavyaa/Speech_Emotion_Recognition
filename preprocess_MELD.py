import torch
import torchaudio
import os
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd

def load_audio(file_path):
    """Load and preprocess audio file from MP4."""
    try:
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_path = file_path.replace('.mp4', '.wav')
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        waveform, sample_rate = torchaudio.load(audio_path)
        os.remove(audio_path)  # Remove the temporary WAV file
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy()
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        return None

def get_all_mp4_files(dataset_path):
    """Recursively get all MP4 files from the dataset directory."""
    mp4_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mp4'):
                full_path = os.path.join(root, file)
                mp4_files.append(full_path)
                print(f"Found MP4 file: {full_path}")  # Debugging print statement
    return mp4_files

def extract_embeddings(dataset_path, save_prefix, labels_df):
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

    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Dataset path: {dataset_path}")  # Debugging print statement

    # Get list of all MP4 files from the dataset directory
    mp4_files = get_all_mp4_files(dataset_path)
    print(f"Found {len(mp4_files)} MP4 files in dataset")
    
    if len(mp4_files) == 0:
        raise ValueError(f"No MP4 files found in {dataset_path}")

    # Define emotion labels mapping
    emotion_mapping = {
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "surprise": 4,
        "fear": 5,
        "disgust": 6
    }

    embeddings = []
    labels = []
    processed_files = []

    for file_path in tqdm(mp4_files, desc="Processing audio files"):
        try:
            # Extract emotion label from the labels dataframe
            filename = os.path.basename(file_path)
            dialogue_id = int(filename.split('_')[0][3:])
            utterance_id = int(filename.split('_')[1][3:].split('.')[0])
            emotion_label = labels_df.loc[(labels_df['Dialogue_ID'] == dialogue_id) & (labels_df['Utterance_ID'] == utterance_id), 'Emotion'].values
            if len(emotion_label) == 0:
                print(f"Skipping {filename}: No emotion label found")
                continue
            emotion = emotion_mapping.get(emotion_label[0], None)
            if emotion is None:
                print(f"Skipping {filename}: Unknown emotion label")
                continue
            
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
    
    # Save new files with specified names
    np.save(os.path.join(save_path, f'embeddings_{save_prefix}_MELD.npy'), embeddings)
    np.save(os.path.join(save_path, f'labels_{save_prefix}_MELD.npy'), labels)
    
    # Save processed files list for reference
    with open(os.path.join(save_path, f'processed_files_{save_prefix}.txt'), 'w') as f:
        for file_path in processed_files:
            f.write(f"{file_path}\n")
    
    print(f"Successfully processed {len(embeddings)} files")
    print(f"Saved embeddings shape: {embeddings.shape}")
    print(f"Saved labels shape: {labels.shape}")
    print(f"Files saved to: {save_path}")
    
    return embeddings, labels

if __name__ == "__main__":
    try:
        # Load labels CSV for train set
        train_labels_csv_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\MELD\train\train_sent_emo.csv"
        train_labels_df = pd.read_csv(train_labels_csv_path)

        # Process train set
        train_dataset_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\MELD\train\train_splits"
        extract_embeddings(train_dataset_path, "train", train_labels_df)

        # Load labels CSV for dev set
        dev_labels_csv_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\MELD\dev_sent_emo.csv"
        dev_labels_df = pd.read_csv(dev_labels_csv_path)

        # Process dev set
        dev_dataset_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\MELD\dev\dev_splits_complete"
        extract_embeddings(dev_dataset_path, "dev", dev_labels_df)

        # Load labels CSV for test set
        test_labels_csv_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\MELD\test_sent_emo.csv"
        test_labels_df = pd.read_csv(test_labels_csv_path)

        # Process test set
        test_dataset_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\MELD\test\output_repeated_splits_test"
        extract_embeddings(test_dataset_path, "test", test_labels_df)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

# Verify files are saved
save_path = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition"
print("Files in directory:")
for file in os.listdir(save_path):
    if file.startswith('embeddings_') or file.startswith('labels_'):
        print(f"- {file}")
