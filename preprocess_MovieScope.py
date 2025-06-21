import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm

def load_audio_segments(file_path, segment_length=10, sr=16000):
    """Load audio and split into segments of segment_length seconds."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()
        total_samples = waveform.shape[0]
        segment_samples = segment_length * sr
        segments = []
        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            segment = waveform[start:end]
            # Pad if segment is shorter than segment_samples
            if segment.shape[0] < segment_samples:
                segment = torch.nn.functional.pad(segment, (0, segment_samples - segment.shape[0]))
            segments.append(segment.numpy())
        return segments
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def get_all_audio_files(dataset_path):
    """Recursively get all .wav files from dataset."""
    audio_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def extract_moviescope_embeddings():
    # Initialize model and feature extractor
    model = Wav2Vec2Model.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        ignore_mismatched_sizes=True,
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset_path = r"D:\Moviescope\audio_wav"
    save_path = r"D:\Moviescope"
    audio_files = get_all_audio_files(dataset_path)
    print(f"Found {len(audio_files)} audio files in Moviescope.")

    # Resume logic
    partial_embeddings_path = os.path.join(save_path, "embeddings_MOVIESCOPE_partial.npy")
    partial_segment_counts_path = os.path.join(save_path, "segment_counts_MOVIESCOPE_partial.npy")
    partial_processed_files_path = os.path.join(save_path, "processed_files_MOVIESCOPE_partial.txt")
    partial_failed_files_path = os.path.join(save_path, "failed_files_MOVIESCOPE_partial.txt")

    if os.path.exists(partial_embeddings_path):
        all_embeddings = list(np.load(partial_embeddings_path, allow_pickle=True))
        segment_counts = list(np.load(partial_segment_counts_path, allow_pickle=True))
        with open(partial_processed_files_path, "r", encoding="utf-8") as f:
            processed_files = [line.strip() for line in f]
        with open(partial_failed_files_path, "r", encoding="utf-8") as f:
            failed_files = [line.strip().split('\t')[0] for line in f]
        print(f"Resuming: {len(processed_files)} files already processed, {len(failed_files)} failed.")
    else:
        all_embeddings = []
        segment_counts = []
        processed_files = []
        failed_files = []

    # Only process files not already done
    files_to_process = [f for f in audio_files if f not in processed_files and f not in failed_files]

    save_every = 100  # Save progress every 100 files

    for idx, file_path in enumerate(tqdm(files_to_process, desc="Processing Moviescope audio files")):
        segments = load_audio_segments(file_path, segment_length=10, sr=16000)  # 10s segments
        file_embeddings = []
        if not segments or all(len(seg) == 0 for seg in segments):
            print(f"SKIP: No valid segments for {file_path}")
            failed_files.append((file_path, "No valid segments"))
            continue
        for seg_idx, segment in enumerate(segments):
            try:
                inputs = feature_extractor(
                    segment, sampling_rate=16000, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                file_embeddings.append(embedding.squeeze())
            except Exception as e:
                print(f"Error extracting embedding for {file_path} segment {seg_idx}: {str(e)}")
                failed_files.append((file_path, f"Segment {seg_idx} embedding error: {str(e)}"))
        if file_embeddings:
            all_embeddings.append(np.stack(file_embeddings))
            segment_counts.append(len(file_embeddings))
            processed_files.append(file_path)
        else:
            print(f"SKIP: No valid embeddings for {file_path}")
            failed_files.append((file_path, "No valid embeddings"))

        # Save progress every N files
        if (idx + 1) % save_every == 0:
            np.save(os.path.join(save_path, "embeddings_MOVIESCOPE_partial.npy"), np.array(all_embeddings, dtype=object))
            np.save(os.path.join(save_path, "segment_counts_MOVIESCOPE_partial.npy"), np.array(segment_counts))
            with open(os.path.join(save_path, "processed_files_MOVIESCOPE_partial.txt"), "w", encoding="utf-8") as f:
                for fname in processed_files:
                    f.write(f"{fname}\n")
            with open(os.path.join(save_path, "failed_files_MOVIESCOPE_partial.txt"), "w", encoding="utf-8") as f:
                for fname, reason in failed_files:
                    f.write(f"{fname}\t{reason}\n")
            print(f"Progress saved at {idx+1} files.")

        # Free up memory
        del file_embeddings, segments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    np.save(os.path.join(save_path, "embeddings_MOVIESCOPE.npy"), np.array(all_embeddings, dtype=object))
    np.save(os.path.join(save_path, "segment_counts_MOVIESCOPE.npy"), np.array(segment_counts))
    with open(os.path.join(save_path, "processed_files_MOVIESCOPE.txt"), "w", encoding="utf-8") as f:
        for fname in processed_files:
            f.write(f"{fname}\n")
    with open(os.path.join(save_path, "failed_files_MOVIESCOPE.txt"), "w", encoding="utf-8") as f:
        for fname, reason in failed_files:
            f.write(f"{fname}\t{reason}\n")

    print(f"Saved embeddings for {len(all_embeddings)} trailers.")
    print(f"Failed to process {len(failed_files)} files. See failed_files_MOVIESCOPE.txt for details.")
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    extract_moviescope_embeddings()