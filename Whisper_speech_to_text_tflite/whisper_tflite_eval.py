
import pyaudio
import wave
import time
import os
import re
import tempfile
from whisper_tflite import WhisperModel
from collections import Counter

# -------------------------
# Accuracy metrics
# -------------------------
_word_re = re.compile(r"[a-z0-9']+")

def tokenize(text: str):
    # Lowercase and keep only word characters (alnum and apostrophes).
    # This ignores punctuation like commas/periods so “activate, steady, then stop.”
    # matches “activate steady then stop”.
    return _word_re.findall(text.lower())

def levenshtein_distance(a_tokens, b_tokens):
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]

def compute_metrics(target_text: str, recognized_text: str):
    tgt_tokens = tokenize(target_text)
    rec_tokens = tokenize(recognized_text)

    tgt_counts = Counter(tgt_tokens)
    rec_counts = Counter(rec_tokens)

    overlap = sum(min(tgt_counts[w], rec_counts[w]) for w in set(tgt_counts) | set(rec_counts))
    precision = overlap / max(1, sum(rec_counts.values()))
    recall = overlap / max(1, sum(tgt_counts.values()))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    dist = levenshtein_distance(tgt_tokens, rec_tokens)
    max_len = max(len(tgt_tokens), len(rec_tokens), 1)
    lev_similarity = 1.0 - (dist / max_len)

    missed = list((tgt_counts - rec_counts).elements())
    extra = list((rec_counts - tgt_counts).elements())

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lev_similarity": lev_similarity,
        "missed": missed,
        "extra": extra
    }

# -------------------------
# AudioTranscriber class
# -------------------------
class AudioTranscriber:
    def __init__(self, model_path="./whisper-tiny-en.tflite"):
        self.model = WhisperModel(model_path)
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.p = pyaudio.PyAudio()
        self.default_target_sentence = "activate steady then stop"

    def list_audio_devices(self):
        print("\n=== Available Audio Input Devices ===")
        devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
                print(f"{i}: {device_info['name']} "
                      f"(Channels: {device_info['maxInputChannels']}, "
                      f"Sample Rate: {int(device_info['defaultSampleRate'])})")
        print("=" * 50)
        return devices

    def select_device(self):
        devices = self.list_audio_devices()
        if not devices:
            print("No audio input devices found!")
            return None
        while True:
            try:
                choice = input(f"\nSelect device index or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                device_index = int(choice)
                for device in devices:
                    if device['index'] == device_index:
                        print(f"Selected: {device['name']}")
                        return device['index']
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number or 'q'.")

    def record_audio(self, device_index, duration=None):
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )

            print(f'\n🎯 Target Sentence: "{self.default_target_sentence}"')
            print("Press Ctrl+C to stop...")
            if duration:
                print(f"Recording for {duration} seconds...")

            self.audio_data = []
            self.is_recording = True
            start_time = time.time()
            while self.is_recording:
                if duration and (time.time() - start_time) >= duration:
                    break
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_data.append(data)
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            self.is_recording = False

    def save_audio_to_file(self, filename):
        if not self.audio_data:
            print("No audio data to save!")
            return False
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_data))
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False

    def transcribe_audio(self, audio_file):
        try:
            print("\nTranscribing audio...")
            segments, _ = self.model.transcribe(audio_file)
            recognized_text = " ".join([seg.text.strip() for seg in segments])
            print("\n=== Transcription Results ===")
            print(recognized_text)
            print("=" * 50)
            return recognized_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def record_and_transcribe(self, device_index, duration=None):
        # Show target before recording
        print(f'\n🎯 Target Sentence: "{self.default_target_sentence}"')
        input("Press Enter to start recording, then speak...\n")

        self.record_audio(device_index, duration)

        if not self.audio_data:
            print("No audio recorded!")
            return
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        try:
            if self.save_audio_to_file(temp_filename):
                recognized_text = self.transcribe_audio(temp_filename)
                if recognized_text is not None:
                    metrics = compute_metrics(self.default_target_sentence, recognized_text)
                    print("\n=== Accuracy Metrics ===")
                    print(f"Target Sentence: {self.default_target_sentence}")
                    print(f"Recognized: {recognized_text}")
                    print(f"Precision: {metrics['precision']*100:.2f}%")
                    print(f"Recall: {metrics['recall']*100:.2f}%")
                    print(f"F1 Score: {metrics['f1']*100:.2f}%")
                    print(f"Levenshtein Similarity: {metrics['lev_similarity']*100:.2f}%")
                    print(f"Missed words: {metrics['missed']}")
                    print(f"Extra words: {metrics['extra']}")
                return recognized_text
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def cleanup(self):
        self.p.terminate()

# -------------------------
# Main Program
# -------------------------
def main():
    transcriber = AudioTranscriber()
    try:
        device_index = transcriber.select_device()
        if device_index is None:
            print("No device selected. Exiting.")
            return
        transcriber.record_and_transcribe(device_index)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        transcriber.cleanup()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()

