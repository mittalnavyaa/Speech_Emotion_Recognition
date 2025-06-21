import os
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_dir, wav_dir):
    os.makedirs(wav_dir, exist_ok=True)
    for filename in os.listdir(mp3_dir):
        if filename.lower().endswith('.mp3'):
            mp3_path = os.path.join(mp3_dir, filename)
            wav_path = os.path.join(wav_dir, os.path.splitext(filename)[0] + '.wav')
            if not os.path.exists(wav_path):
                try:
                    audio = AudioSegment.from_file(mp3_path, format="mp3")
                    audio.export(wav_path, format="wav")
                    print(f"Converted: {filename}")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    mp3_dir = r"D:\Moviescope\audio"
    wav_dir = r"D:\Moviescope\audio_wav"
    convert_mp3_to_wav(mp3_dir, wav_dir)