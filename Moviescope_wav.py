import os
import torchaudio

def convert_and_cleanup_mp3_to_wav(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                mp3_path = os.path.join(subdir, file)
                wav_path = os.path.splitext(mp3_path)[0] + '.wav'
                # Only convert if .wav does not exist
                if not os.path.exists(wav_path):
                    try:
                        waveform, sample_rate = torchaudio.load(mp3_path)
                        torchaudio.save(wav_path, waveform, sample_rate)
                        print(f"Converted: {mp3_path} -> {wav_path}")
                    except Exception as e:
                        print(f"Error converting {mp3_path}: {str(e)}")
                # Delete the mp3 file regardless of conversion success
                try:
                    os.remove(mp3_path)
                    print(f"Deleted: {mp3_path}")
                except Exception as e:
                    print(f"Error deleting {mp3_path}: {str(e)}")

if __name__ == "__main__":
    moviescope_dir = r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\Moviescope\Data"
    convert_and_cleanup_mp3_to_wav(moviescope_dir)