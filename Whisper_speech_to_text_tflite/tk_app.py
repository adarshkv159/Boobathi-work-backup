import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import pyaudio
import wave
import threading
import time
import os
import tempfile
from whisper_tflite import WhisperModel

class AudioTranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Audio Transcriber")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Initialize audio components
        self.model = None
        self.p = pyaudio.PyAudio()
        self.is_recording = False
        self.is_continuous = False
        self.audio_data = []
        self.recording_thread = None
        self.continuous_thread = None
        
        # Audio settings
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.selected_device_index = None
        
        self.setup_ui()
        self.load_audio_devices()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎙️ Whisper Audio Transcriber", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Model Settings", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.model_path_var = tk.StringVar(value="./whisper-tiny-en.tflite")
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        self.model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.browse_button = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        self.browse_button.grid(row=0, column=2, sticky=tk.W)
        
        self.load_model_button = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)
        
        self.model_status_label = ttk.Label(model_frame, text="Model not loaded", foreground="red")
        self.model_status_label.grid(row=1, column=1, pady=(10, 0), sticky=tk.W, padx=(10, 0))
        
        # Device selection section
        device_frame = ttk.LabelFrame(main_frame, text="Audio Device", padding="10")
        device_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        device_frame.columnconfigure(1, weight=1)
        
        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                        state="readonly", width=60)
        self.device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_selected)
        
        self.refresh_devices_button = ttk.Button(device_frame, text="Refresh", 
                                               command=self.load_audio_devices)
        self.refresh_devices_button.grid(row=0, column=2, sticky=tk.W)
        
        # Recording controls section
        controls_frame = ttk.LabelFrame(main_frame, text="Recording Controls", padding="10")
        controls_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Duration setting
        duration_frame = ttk.Frame(controls_frame)
        duration_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(duration_frame, text="Duration (seconds):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.duration_var = tk.StringVar(value="5")
        self.duration_spinbox = ttk.Spinbox(duration_frame, from_=1, to=60, 
                                          textvariable=self.duration_var, width=10)
        self.duration_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(duration_frame, text="Chunk size (continuous):").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.chunk_duration_var = tk.StringVar(value="5")
        self.chunk_spinbox = ttk.Spinbox(duration_frame, from_=3, to=15, 
                                       textvariable=self.chunk_duration_var, width=10)
        self.chunk_spinbox.grid(row=0, column=3, sticky=tk.W)
        
        # Control buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        self.record_button = ttk.Button(buttons_frame, text="🔴 Start Recording", 
                                       command=self.start_recording, width=15)
        self.record_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(buttons_frame, text="⏹️ Stop Recording", 
                                     command=self.stop_recording, state=tk.DISABLED, width=15)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.timed_record_button = ttk.Button(buttons_frame, text="⏲️ Timed Recording", 
                                            command=self.start_timed_recording, width=15)
        self.timed_record_button.grid(row=0, column=2, padx=(0, 10))
        
        self.continuous_button = ttk.Button(buttons_frame, text="🔄 Start Continuous", 
                                          command=self.toggle_continuous, width=15)
        self.continuous_button.grid(row=0, column=3)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready to record", foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        status_frame.columnconfigure(0, weight=1)
        
        # File operations section
        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding="10")
        file_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.transcribe_file_button = ttk.Button(file_frame, text="📁 Transcribe Audio File", 
                                               command=self.transcribe_file)
        self.transcribe_file_button.grid(row=0, column=0, padx=(0, 10))
        
        self.save_transcription_button = ttk.Button(file_frame, text="💾 Save Transcription", 
                                                   command=self.save_transcription)
        self.save_transcription_button.grid(row=0, column=1, padx=(0, 10))
        
        self.clear_button = ttk.Button(file_frame, text="🗑️ Clear Results", 
                                     command=self.clear_results)
        self.clear_button.grid(row=0, column=2)
        
        # Transcription results section
        results_frame = ttk.LabelFrame(main_frame, text="Transcription Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     width=80, height=20, font=("Consolas", 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Whisper TFLite Model",
            filetypes=[("TFLite files", "*.tflite"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """Load the Whisper model"""
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            self.update_status("Loading model...", "orange")
            self.progress.start()
            
            # Load model in a separate thread to avoid blocking UI
            def load_model_thread():
                try:
                    self.model = WhisperModel(model_path)
                    self.root.after(0, lambda: self.on_model_loaded(True))
                except Exception as e:
                    self.root.after(0, lambda: self.on_model_loaded(False, str(e)))
            
            threading.Thread(target=load_model_thread, daemon=True).start()
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.update_status("Model load failed", "red")
    
    def on_model_loaded(self, success, error=None):
        """Handle model loading completion"""
        self.progress.stop()
        if success:
            self.model_status_label.config(text="Model loaded successfully", foreground="green")
            self.update_status("Model loaded - Ready to record", "green")
        else:
            self.model_status_label.config(text=f"Model load failed", foreground="red")
            self.update_status(f"Model load failed: {error}", "red")
            messagebox.showerror("Model Error", f"Failed to load model: {error}")
    
    def load_audio_devices(self):
        """Load available audio input devices"""
        try:
            devices = []
            device_list = []
            
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    device_name = f"{i}: {device_info['name']} ({device_info['maxInputChannels']} ch, {int(device_info['defaultSampleRate'])}Hz)"
                    devices.append((i, device_name))
                    device_list.append(device_name)
            
            self.device_combo['values'] = device_list
            self.device_data = devices
            
            if device_list:
                self.device_combo.current(0)
                self.on_device_selected()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio devices: {e}")
    
    def on_device_selected(self, event=None):
        """Handle device selection"""
        if self.device_combo.current() >= 0:
            self.selected_device_index = self.device_data[self.device_combo.current()][0]
            self.update_status(f"Device selected: {self.device_var.get()}", "green")
    
    def update_status(self, message, color="black"):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
    
    def validate_ready(self):
        """Check if system is ready for recording"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return False
        
        if self.selected_device_index is None:
            messagebox.showerror("Error", "Please select an audio device")
            return False
        
        return True
    
    def start_recording(self):
        """Start manual recording"""
        if not self.validate_ready():
            return
        
        if self.is_recording:
            return
        
        self.is_recording = True
        self.update_controls_state()
        self.update_status("Recording... Click Stop to finish", "red")
        
        self.recording_thread = threading.Thread(target=self.record_audio_manual, daemon=True)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop manual recording"""
        self.is_recording = False
        self.update_status("Stopping recording...", "orange")
    
    def start_timed_recording(self):
        """Start timed recording"""
        if not self.validate_ready():
            return
        
        try:
            duration = float(self.duration_var.get())
            if duration <= 0:
                raise ValueError("Duration must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid duration")
            return
        
        self.is_recording = True
        self.update_controls_state()
        self.update_status(f"Recording for {duration} seconds...", "red")
        
        self.recording_thread = threading.Thread(target=self.record_audio_timed, 
                                                args=(duration,), daemon=True)
        self.recording_thread.start()
    
    def toggle_continuous(self):
        """Toggle continuous transcription"""
        if not self.validate_ready():
            return
        
        if self.is_continuous:
            self.stop_continuous()
        else:
            self.start_continuous()
    
    def start_continuous(self):
        """Start continuous transcription"""
        try:
            chunk_duration = float(self.chunk_duration_var.get())
            if chunk_duration < 1:
                raise ValueError("Chunk duration must be at least 1 second")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid chunk duration")
            return
        
        self.is_continuous = True
        self.continuous_button.config(text="⏹️ Stop Continuous")
        self.update_status("Starting continuous transcription...", "blue")
        
        self.continuous_thread = threading.Thread(target=self.continuous_transcription, 
                                                 args=(chunk_duration,), daemon=True)
        self.continuous_thread.start()
    
    def stop_continuous(self):
        """Stop continuous transcription"""
        self.is_continuous = False
        self.continuous_button.config(text="🔄 Start Continuous")
        self.update_status("Stopping continuous transcription...", "orange")
    
    def record_audio_manual(self):
        """Record audio manually until stopped"""
        self.audio_data = []
        stream = None
        
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break
                    
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Recording Error", f"Failed to record: {e}"))
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            self.is_recording = False
            self.root.after(0, self.on_recording_finished)
    
    def record_audio_timed(self, duration):
        """Record audio for specified duration"""
        self.audio_data = []
        stream = None
        
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            start_time = time.time()
            while self.is_recording and (time.time() - start_time) < duration:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break
                    
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Recording Error", f"Failed to record: {e}"))
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            self.is_recording = False
            self.root.after(0, self.on_recording_finished)
    
    def continuous_transcription(self, chunk_duration):
        """Continuously record and transcribe"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, 
                                                           f"\n=== Started Continuous Transcription (chunks: {chunk_duration}s) ===\n"))
        
        while self.is_continuous:
            # Record chunk
            self.audio_data = []
            stream = None
            
            try:
                stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.selected_device_index,
                    frames_per_buffer=self.chunk_size
                )
                
                start_time = time.time()
                self.root.after(0, lambda: self.update_status(f"Recording chunk ({chunk_duration}s)...", "red"))
                
                while self.is_continuous and (time.time() - start_time) < chunk_duration:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)
                    
            except Exception as e:
                self.root.after(0, lambda: self.append_result(f"❌ Recording error: {e}\n"))
                break
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()
            
            if not self.is_continuous:
                break
                
            # Transcribe chunk
            if self.audio_data:
                self.root.after(0, lambda: self.update_status("Transcribing chunk...", "blue"))
                segments = self.transcribe_audio_data()
                
                if segments:
                    full_text = " ".join([segment.text.strip() for segment in segments])
                    if full_text.strip():
                        timestamp = time.strftime("%H:%M:%S")
                        self.root.after(0, lambda t=timestamp, txt=full_text: 
                                      self.append_result(f"[{t}] 💬 {txt}\n"))
                    else:
                        timestamp = time.strftime("%H:%M:%S")
                        self.root.after(0, lambda t=timestamp: 
                                      self.append_result(f"[{t}] 🔇 No speech detected\n"))
                else:
                    timestamp = time.strftime("%H:%M:%S")
                    self.root.after(0, lambda t=timestamp: 
                                  self.append_result(f"[{t}] ❌ Transcription failed\n"))
            
            if self.is_continuous:
                time.sleep(0.5)  # Small pause between chunks
        
        self.root.after(0, lambda: self.update_status("Continuous transcription stopped", "green"))
    
    def on_recording_finished(self):
        """Handle recording completion"""
        self.update_controls_state()
        self.update_status("Processing recording...", "blue")
        self.progress.start()
        
        if self.audio_data:
            # Transcribe in separate thread
            threading.Thread(target=self.process_recorded_audio, daemon=True).start()
        else:
            self.progress.stop()
            self.update_status("No audio recorded", "orange")
    
    def process_recorded_audio(self):
        """Process and transcribe recorded audio"""
        try:
            segments = self.transcribe_audio_data()
            self.root.after(0, lambda: self.on_transcription_complete(segments))
        except Exception as e:
            self.root.after(0, lambda: self.on_transcription_error(str(e)))
    
    def transcribe_audio_data(self):
        """Transcribe current audio data"""
        if not self.audio_data or not self.model:
            return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # Save audio data
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_data))
            
            # Transcribe
            segments, _ = self.model.transcribe(temp_filename)
            return segments
            
        except Exception as e:
            raise e
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def on_transcription_complete(self, segments):
        """Handle successful transcription"""
        self.progress.stop()
        self.update_status("Transcription complete", "green")
        
        if segments:
            self.results_text.insert(tk.END, f"\n=== Transcription Results ({time.strftime('%H:%M:%S')}) ===\n")
            for segment in segments:
                result = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                self.results_text.insert(tk.END, result)
            self.results_text.insert(tk.END, "=" * 60 + "\n")
        else:
            self.results_text.insert(tk.END, f"\n[{time.strftime('%H:%M:%S')}] No speech detected\n")
        
        # Auto-scroll to bottom
        self.results_text.see(tk.END)
    
    def on_transcription_error(self, error):
        """Handle transcription error"""
        self.progress.stop()
        self.update_status("Transcription failed", "red")
        messagebox.showerror("Transcription Error", f"Failed to transcribe: {error}")
    
    def append_result(self, text):
        """Append text to results"""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def transcribe_file(self):
        """Transcribe an audio file"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return
        
        self.update_status("Transcribing file...", "blue")
        self.progress.start()
        
        def transcribe_thread():
            try:
                segments, _ = self.model.transcribe(filename)
                self.root.after(0, lambda: self.on_file_transcription_complete(segments, filename))
            except Exception as e:
                self.root.after(0, lambda: self.on_transcription_error(str(e)))
        
        threading.Thread(target=transcribe_thread, daemon=True).start()
    
    def on_file_transcription_complete(self, segments, filename):
        """Handle file transcription completion"""
        self.progress.stop()
        self.update_status("File transcription complete", "green")
        
        self.results_text.insert(tk.END, f"\n=== File Transcription: {os.path.basename(filename)} ===\n")
        if segments:
            for segment in segments:
                result = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                self.results_text.insert(tk.END, result)
        else:
            self.results_text.insert(tk.END, "No speech detected\n")
        
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.see(tk.END)
    
    def save_transcription(self):
        """Save transcription results to file"""
        content = self.results_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No transcription results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Transcription",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Transcription saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def clear_results(self):
        """Clear transcription results"""
        self.results_text.delete("1.0", tk.END)
        self.update_status("Results cleared", "green")
    
    def update_controls_state(self):
        """Update control button states"""
        if self.is_recording:
            self.record_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.timed_record_button.config(state=tk.DISABLED)
            self.continuous_button.config(state=tk.DISABLED)
        else:
            self.record_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.timed_record_button.config(state=tk.NORMAL)
            if not self.is_continuous:
                self.continuous_button.config(state=tk.NORMAL)
    
    def on_closing(self):
        """Handle application closing"""
        # Stop any ongoing operations
        self.is_recording = False
        self.is_continuous = False
        
        # Wait a moment for threads to finish
        time.sleep(0.5)
        
        # Clean up PyAudio
        try:
            self.p.terminate()
        except:
            pass
        
        self.root.quit()
        self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set application icon (if you have one)
    try:
        # root.iconbitmap("icon.ico")  # Uncomment if you have an icon file
        pass
    except:
        pass
    
    app = AudioTranscriberGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("Application closed")

if __name__ == "__main__":
    main()
