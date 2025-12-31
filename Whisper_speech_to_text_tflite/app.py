import pyaudio
import wave
import threading
import time
import os
import tempfile
from whisper_tflite import WhisperModel

class AudioTranscriber:
    def __init__(self, model_path="./whisper-tiny-en.tflite"):
        self.model = WhisperModel(model_path)
        self.is_recording = False
        self.audio_data = []
        self.sample_rate =  44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.p = pyaudio.PyAudio()
        
    def list_audio_devices(self):
        """List all available audio input devices"""
        print("\n=== Available Audio Input Devices ===")
        devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            # Check if device has input channels
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
        """Let user select an audio input device"""
        devices = self.list_audio_devices()
        if not devices:
            print("No audio input devices found!")
            return None
            
        while True:
            try:
                choice = input(f"\nSelect device (0-{len(devices)-1}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                    
                device_index = int(choice)
                # Find the actual device index from our filtered list
                selected_device = None
                for device in devices:
                    if device['index'] == device_index:
                        selected_device = device
                        break
                        
                if selected_device:
                    print(f"Selected: {selected_device['name']}")
                    return selected_device['index']
                else:
                    print("Invalid selection. Please try again.")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
    
    def record_audio(self, device_index, duration=None):
        """Record audio from selected device"""
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"\nRecording... Press Ctrl+C to stop")
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
        """Save recorded audio to WAV file"""
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
        """Transcribe audio file using Whisper"""
        try:
            print("\nTranscribing audio...")
            segments, _ = self.model.transcribe(audio_file)
            
            print("\n=== Transcription Results ===")
            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            print("=" * 50)
            
            return segments
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def record_and_transcribe(self, device_index, duration=None):
        """Record audio and transcribe it"""
        # Record audio
        self.record_audio(device_index, duration)
        
        if not self.audio_data:
            print("No audio recorded!")
            return
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            if self.save_audio_to_file(temp_filename):
                # Transcribe
                segments = self.transcribe_audio(temp_filename)
                return segments
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def continuous_transcription(self, device_index, chunk_duration=5):
        """Continuously record and transcribe audio in chunks"""
        print(f"\nStarting continuous transcription (Press Ctrl+C to stop)")
        print(f"Recording in {chunk_duration}-second chunks...")
        
        try:
            while True:
                print(f"\n--- Recording chunk ({chunk_duration}s) ---")
                segments = self.record_and_transcribe(device_index, chunk_duration)
                
                if segments:
                    # Combine all text from segments
                    full_text = " ".join([segment.text.strip() for segment in segments])
                    if full_text.strip():
                        print(f"💬 Transcribed: {full_text}")
                    else:
                        print("🔇 No speech detected")
                else:
                    print("❌ Transcription failed")
                    
                time.sleep(0.5)  # Small pause between chunks
                
        except KeyboardInterrupt:
            print("\n\nStopping continuous transcription...")
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        self.p.terminate()

def main():
    transcriber = AudioTranscriber()
    
    try:
        # Select audio device
        device_index = transcriber.select_device()
        if device_index is None:
            print("No device selected. Exiting.")
            return
        
        # Choose mode
        print("\n=== Recording Mode ===")
        print("1. Record once and transcribe")
        print("2. Record for specific duration")
        print("3. Continuous transcription")
        
        while True:
            choice = input("\nSelect mode (1-3) or 'q' to quit: ").strip()
            
            if choice == '1':
                print("\nPress Enter to start recording, then Ctrl+C to stop...")
                input()
                transcriber.record_and_transcribe(device_index)
                break
                
            elif choice == '2':
                try:
                    duration = float(input("Enter recording duration (seconds): "))
                    transcriber.record_and_transcribe(device_index, duration)
                    break
                except ValueError:
                    print("Please enter a valid number.")
                    
            elif choice == '3':
                try:
                    chunk_duration = input("Enter chunk duration (default 5s): ").strip()
                    chunk_duration = float(chunk_duration) if chunk_duration else 5.0
                    transcriber.continuous_transcription(device_index, chunk_duration)
                    break
                except ValueError:
                    print("Please enter a valid number.")
                    
            elif choice.lower() == 'q':
                break
                
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        transcriber.cleanup()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
