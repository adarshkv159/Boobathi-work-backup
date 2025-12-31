import sys
import pyaudio
import wave
import threading
import time
import os
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QPushButton, QTextEdit, QComboBox,
                                QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
                                QTabWidget, QProgressBar, QMessageBox, QFileDialog)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from whisper_tflite import WhisperModel


class AudioRecorderThread(QThread):
    recording_started = Signal()
    recording_stopped = Signal()
    transcription_started = Signal()
    transcription_complete = Signal(str)
    error_occurred = Signal(str)
    status_update = Signal(str)
    
    def __init__(self, model, device_index, duration=None, continuous=False, 
                 chunk_duration=5):
        super().__init__()
        # Create dedicated PyAudio instance per thread
        self.model = model
        self.device_index = device_index
        self.duration = duration
        self.continuous = continuous
        self.chunk_duration = chunk_duration
        self.should_stop = False
        self._is_running = False
        
        # Thread-local audio settings
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_data = []
        self.is_recording = False
        
        # PyAudio instance will be created in run() to avoid cross-thread issues
        self.p = None
    
    def run(self):
        self._is_running = True
        
        # Create PyAudio in the worker thread
        self.p = pyaudio.PyAudio()
        
        try:
            if self.continuous:
                self.run_continuous()
            else:
                self.run_single()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Proper PyAudio cleanup
            if self.p is not None:
                try:
                    self.p.terminate()
                except Exception as e:
                    print(f"PyAudio cleanup warning: {e}")
                self.p = None
            self._is_running = False
    
    def record_audio(self, duration=None):
        """Record audio in the worker thread"""
        stream = None
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.audio_data = []
            self.is_recording = True
            start_time = time.time()
            
            while self.is_recording and not self.should_stop:
                if duration and (time.time() - start_time) >= duration:
                    break
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_data.append(data)
                except Exception as e:
                    print(f"Error reading audio chunk: {e}")
                    break
                    
        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")
        finally:
            # Guaranteed stream cleanup
            if stream is not None:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Stream cleanup warning: {e}")
            self.is_recording = False
    
    def save_audio_to_file(self, filename):
        """Save recorded audio to WAV file"""
        if not self.audio_data:
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
    
    def run_single(self):
        try:
            self.recording_started.emit()
            self.status_update.emit("Recording audio...")
            
            # Record audio
            self.record_audio(self.duration)
            
            if not self.audio_data:
                self.error_occurred.emit("No audio recorded!")
                self.recording_stopped.emit()
                return
            
            # Signal recording stopped
            self.recording_stopped.emit()
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_filename = tmp_file.name
            
            try:
                # Save audio
                if self.save_audio_to_file(temp_filename):
                    self.transcription_started.emit()
                    self.status_update.emit("Transcribing audio... Please wait")
                    
                    # Transcribe
                    segments, _ = self.model.transcribe(temp_filename)
                    segments_list = list(segments)
                    
                    if segments_list:
                        output_text = self.format_segments(segments_list)
                        self.transcription_complete.emit(output_text)
                    else:
                        self.error_occurred.emit("Transcription failed!")
                else:
                    self.error_occurred.emit("Failed to save audio file!")
            finally:
                # Clean up temp file
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception as e:
                        print(f"Warning: Could not remove temp file: {e}")
                        
        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")
    
    def run_continuous(self):
        try:
            self.status_update.emit("Starting continuous transcription...")
            
            while not self.should_stop:
                self.recording_started.emit()
                self.status_update.emit(f"Recording chunk ({self.chunk_duration}s)...")
                
                self.record_audio(self.chunk_duration)
                
                if not self.audio_data:
                    time.sleep(0.5)
                    continue
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                
                try:
                    if self.save_audio_to_file(temp_filename):
                        self.transcription_started.emit()
                        self.status_update.emit("Transcribing...")
                        
                        segments, _ = self.model.transcribe(temp_filename)
                        segments_list = list(segments)
                        
                        if segments_list:
                            full_text = " ".join([segment.text.strip() for segment in segments_list])
                            if full_text.strip():
                                timestamp = time.strftime("%H:%M:%S")
                                output = f"[{timestamp}] {full_text}"
                                self.transcription_complete.emit(output)
                finally:
                    if os.path.exists(temp_filename):
                        try:
                            os.remove(temp_filename)
                        except:
                            pass
                
                self.recording_stopped.emit()
                time.sleep(0.5)
                
        except Exception as e:
            self.error_occurred.emit(f"Continuous recording error: {str(e)}")
    
    def format_segments(self, segments):
        """Convert segment objects to formatted string"""
        output_text = ""
        for segment in segments:
            output_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
        return output_text
    
    def stop(self):
        self.should_stop = True
        self.is_recording = False


class AudioTranscriber:
    """Lightweight transcriber - only holds the model"""
    def __init__(self, model_path="./whisper-tiny-en.tflite"):
        self.model = WhisperModel(model_path)
        # Each thread creates its own PyAudio instance
    
    def list_audio_devices(self):
        """List available audio devices (creates temporary PyAudio instance)"""
        p = pyaudio.PyAudio()
        devices = []
        try:
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        devices.append({
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': int(device_info['defaultSampleRate'])
                        })
                except Exception as e:
                    print(f"Error getting device {i}: {e}")
        finally:
            p.terminate()
        return devices
    
    def transcribe_file(self, audio_file):
        """Transcribe audio file (for file tab)"""
        try:
            segments, _ = self.model.transcribe(audio_file)
            return list(segments)
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None


class WhisperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcriber = AudioTranscriber()
        self.recorder_thread = None
        self.selected_device_index = None
        self.transcription_timer = QTimer()
        self.transcription_start_time = 0
        self.animation_timer = QTimer()
        self.animation_state = 0
        
        self.setWindowTitle("Whisper Audio Transcriber")
        self.setGeometry(100, 100, 900, 700)
        self.setStyle(QApplication.setStyle('Fusion'))
        self.init_ui()
        self.load_audio_devices()
        
        # Setup timers
        self.transcription_timer.timeout.connect(self.update_transcription_time)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.setInterval(500)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Device Selection Group
        device_group = QGroupBox("Audio Device Configuration")
        device_layout = QVBoxLayout()
        device_select_layout = QHBoxLayout()
        device_select_layout.addWidget(QLabel("Select Audio Device:"))
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self.on_device_selected)
        device_select_layout.addWidget(self.device_combo)
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_audio_devices)
        device_select_layout.addWidget(self.refresh_btn)
        device_layout.addLayout(device_select_layout)
        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)
        
        # Tab Widget
        self.tab_widget = QTabWidget()
        
        # Tab 1: Single Recording
        single_tab = QWidget()
        single_layout = QVBoxLayout()
        
        single_duration_layout = QHBoxLayout()
        single_duration_layout.addWidget(QLabel("Recording Duration (seconds):"))
        self.single_duration_spin = QSpinBox()
        self.single_duration_spin.setRange(1, 300)
        self.single_duration_spin.setValue(10)
        single_duration_layout.addWidget(self.single_duration_spin)
        single_duration_layout.addStretch()
        single_layout.addLayout(single_duration_layout)
        
        self.record_btn = QPushButton("🎤 Start Recording")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.clicked.connect(self.start_single_recording)
        single_layout.addWidget(self.record_btn)
        
        self.stop_btn = QPushButton("⏹ Stop Recording")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        single_layout.addWidget(self.stop_btn)
        
        transcription_info_layout = QHBoxLayout()
        self.transcription_status_label = QLabel("")
        self.transcription_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        transcription_info_layout.addWidget(self.transcription_status_label)
        self.transcription_timer_label = QLabel("")
        self.transcription_timer_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        transcription_info_layout.addWidget(self.transcription_timer_label)
        transcription_info_layout.addStretch()
        single_layout.addLayout(transcription_info_layout)
        
        single_layout.addWidget(QLabel("Transcription Output:"))
        self.single_output = QTextEdit()
        self.single_output.setReadOnly(True)
        self.single_output.setFont(QFont("Courier", 10))
        single_layout.addWidget(self.single_output)
        
        single_tab.setLayout(single_layout)
        self.tab_widget.addTab(single_tab, "Single Recording")
        
        # Tab 2: Continuous Recording
        continuous_tab = QWidget()
        continuous_layout = QVBoxLayout()
        
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk Duration (seconds):"))
        self.chunk_duration_spin = QDoubleSpinBox()
        self.chunk_duration_spin.setRange(1.0, 30.0)
        self.chunk_duration_spin.setValue(5.0)
        self.chunk_duration_spin.setSingleStep(0.5)
        chunk_layout.addWidget(self.chunk_duration_spin)
        chunk_layout.addStretch()
        continuous_layout.addLayout(chunk_layout)
        
        self.continuous_start_btn = QPushButton("▶ Start Continuous Transcription")
        self.continuous_start_btn.setMinimumHeight(50)
        self.continuous_start_btn.clicked.connect(self.start_continuous_recording)
        continuous_layout.addWidget(self.continuous_start_btn)
        
        self.continuous_stop_btn = QPushButton("⏹ Stop Continuous Transcription")
        self.continuous_stop_btn.setMinimumHeight(50)
        self.continuous_stop_btn.setEnabled(False)
        self.continuous_stop_btn.clicked.connect(self.stop_recording)
        continuous_layout.addWidget(self.continuous_stop_btn)
        
        continuous_info_layout = QHBoxLayout()
        self.continuous_status_label = QLabel("")
        self.continuous_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        continuous_info_layout.addWidget(self.continuous_status_label)
        self.continuous_timer_label = QLabel("")
        self.continuous_timer_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        continuous_info_layout.addWidget(self.continuous_timer_label)
        continuous_info_layout.addStretch()
        continuous_layout.addLayout(continuous_info_layout)
        
        continuous_layout.addWidget(QLabel("Live Transcription:"))
        self.continuous_output = QTextEdit()
        self.continuous_output.setReadOnly(True)
        self.continuous_output.setFont(QFont("Courier", 10))
        continuous_layout.addWidget(self.continuous_output)
        
        self.clear_continuous_btn = QPushButton("Clear Output")
        self.clear_continuous_btn.clicked.connect(lambda: self.continuous_output.clear())
        continuous_layout.addWidget(self.clear_continuous_btn)
        
        continuous_tab.setLayout(continuous_layout)
        self.tab_widget.addTab(continuous_tab, "Continuous Transcription")
        
        # Tab 3: File Transcription
        file_tab = QWidget()
        file_layout = QVBoxLayout()
        
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        file_select_layout.addWidget(self.file_path_label)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_audio_file)
        file_select_layout.addWidget(self.browse_btn)
        file_layout.addLayout(file_select_layout)
        
        self.transcribe_file_btn = QPushButton("📝 Transcribe File")
        self.transcribe_file_btn.setMinimumHeight(50)
        self.transcribe_file_btn.setEnabled(False)
        self.transcribe_file_btn.clicked.connect(self.transcribe_file)
        file_layout.addWidget(self.transcribe_file_btn)
        
        file_info_layout = QHBoxLayout()
        self.file_status_label = QLabel("")
        self.file_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        file_info_layout.addWidget(self.file_status_label)
        self.file_timer_label = QLabel("")
        self.file_timer_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        file_info_layout.addWidget(self.file_timer_label)
        file_info_layout.addStretch()
        file_layout.addLayout(file_info_layout)
        
        file_layout.addWidget(QLabel("Transcription Output:"))
        self.file_output = QTextEdit()
        self.file_output.setReadOnly(True)
        self.file_output.setFont(QFont("Courier", 10))
        file_layout.addWidget(self.file_output)
        
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save Transcription")
        self.save_btn.clicked.connect(self.save_transcription)
        save_layout.addWidget(self.save_btn)
        save_layout.addStretch()
        file_layout.addLayout(save_layout)
        
        file_tab.setLayout(file_layout)
        self.tab_widget.addTab(file_tab, "File Transcription")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status Bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        main_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
    
    def load_audio_devices(self):
        self.device_combo.clear()
        devices = self.transcriber.list_audio_devices()
        
        if not devices:
            QMessageBox.warning(self, "No Devices", "No audio input devices found!")
            return
        
        for device in devices:
            device_text = f"{device['index']}: {device['name']} ({device['channels']} ch, {device['sample_rate']} Hz)"
            self.device_combo.addItem(device_text, device['index'])
    
    def on_device_selected(self, index):
        if index >= 0:
            self.selected_device_index = self.device_combo.itemData(index)
            self.status_label.setText(f"Selected: {self.device_combo.currentText()}")
    
    def start_single_recording(self):
        if self.selected_device_index is None:
            QMessageBox.warning(self, "No Device", "Please select an audio device first!")
            return
        
        # Check if thread is still running from previous operation
        if self.recorder_thread is not None:
            try:
                if self.recorder_thread.isRunning():
                    QMessageBox.warning(self, "Busy", "Please wait for current operation to finish!")
                    return
            except RuntimeError:
                # C++ object deleted - safe to proceed
                pass
            self.recorder_thread = None
        
        duration = self.single_duration_spin.value()
        self.single_output.clear()
        self.transcription_status_label.setText("")
        self.transcription_timer_label.setText("")
        
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        
        # Create new recorder thread with model reference
        self.recorder_thread = AudioRecorderThread(
            self.transcriber.model,
            self.selected_device_index,
            duration=duration
        )
        
        # Connect signals
        self.recorder_thread.status_update.connect(self.update_status)
        self.recorder_thread.transcription_started.connect(self.on_transcription_started)
        self.recorder_thread.transcription_complete.connect(self.on_single_transcription_complete)
        self.recorder_thread.error_occurred.connect(self.on_error)
        self.recorder_thread.recording_stopped.connect(self.on_recording_stopped)
        self.recorder_thread.finished.connect(self.on_thread_finished)
        self.recorder_thread.finished.connect(self.recorder_thread.deleteLater)
        
        self.recorder_thread.start()
    
    def start_continuous_recording(self):
        if self.selected_device_index is None:
            QMessageBox.warning(self, "No Device", "Please select an audio device first!")
            return
        
        # Check if thread is still running from previous operation
        if self.recorder_thread is not None:
            try:
                if self.recorder_thread.isRunning():
                    QMessageBox.warning(self, "Busy", "Please wait for current operation to finish!")
                    return
            except RuntimeError:
                # C++ object deleted - safe to proceed
                pass
            self.recorder_thread = None
        
        chunk_duration = self.chunk_duration_spin.value()
        self.continuous_status_label.setText("")
        self.continuous_timer_label.setText("")
        
        self.continuous_start_btn.setEnabled(False)
        self.continuous_stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        
        # Create new recorder thread
        self.recorder_thread = AudioRecorderThread(
            self.transcriber.model,
            self.selected_device_index,
            continuous=True,
            chunk_duration=chunk_duration
        )
        
        # Connect signals
        self.recorder_thread.status_update.connect(self.update_status)
        self.recorder_thread.transcription_started.connect(self.on_continuous_transcription_started)
        self.recorder_thread.transcription_complete.connect(self.on_continuous_transcription_complete)
        self.recorder_thread.error_occurred.connect(self.on_error)
        self.recorder_thread.finished.connect(self.on_thread_finished)
        self.recorder_thread.finished.connect(self.recorder_thread.deleteLater)
        
        self.recorder_thread.start()
    
    def stop_recording(self):
        if self.recorder_thread is not None:
            try:
                if self.recorder_thread.isRunning():
                    self.status_label.setText("Stopping... Please wait")
                    self.stop_btn.setEnabled(False)
                    self.continuous_stop_btn.setEnabled(False)
                    
                    self.recorder_thread.stop()
                    
                    self.transcription_timer.stop()
                    self.animation_timer.stop()
            except RuntimeError:
                # C++ object already deleted
                pass
    
    def on_recording_stopped(self):
        """Called when recording phase stops (before transcription)"""
        pass
    
    def on_thread_finished(self):
        """Called when thread completely finishes"""
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.continuous_start_btn.setEnabled(True)
        self.continuous_stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.transcription_timer.stop()
        self.animation_timer.stop()
        
        # Clear status labels after a delay
        QTimer.singleShot(3000, self.clear_status_labels)
    
    def clear_status_labels(self):
        """Clear status labels after completion"""
        if self.recorder_thread is None or not self.recorder_thread.isRunning():
            self.transcription_status_label.setText("")
            self.transcription_timer_label.setText("")
            self.continuous_status_label.setText("")
            self.continuous_timer_label.setText("")
            self.file_status_label.setText("")
            self.file_timer_label.setText("")
    
    def on_transcription_started(self):
        """Called when transcription begins for single recording"""
        self.transcription_start_time = time.time()
        self.transcription_timer.start(100)
        self.animation_timer.start()
        self.animation_state = 0
        self.transcription_status_label.setText("🔄 Transcribing")
    
    def on_continuous_transcription_started(self):
        """Called when transcription begins for continuous mode"""
        self.transcription_start_time = time.time()
        self.transcription_timer.start(100)
        self.animation_timer.start()
        self.animation_state = 0
        self.continuous_status_label.setText("🔄 Transcribing")
    
    def update_transcription_time(self):
        """Update the transcription timer display"""
        elapsed = time.time() - self.transcription_start_time
        current_tab = self.tab_widget.currentIndex()
        
        time_text = f"⏱ {elapsed:.1f}s"
        
        if current_tab == 0:
            self.transcription_timer_label.setText(time_text)
        elif current_tab == 1:
            self.continuous_timer_label.setText(time_text)
        else:
            self.file_timer_label.setText(time_text)
    
    def update_animation(self):
        """Update the animated status indicator"""
        animations = ["🔄", "🔃", "⟳", "⟲"]
        self.animation_state = (self.animation_state + 1) % len(animations)
        animation_char = animations[self.animation_state]
        
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:
            self.transcription_status_label.setText(f"{animation_char} Transcribing")
        elif current_tab == 1:
            self.continuous_status_label.setText(f"{animation_char} Transcribing")
        else:
            self.file_status_label.setText(f"{animation_char} Transcribing")
    
    def on_single_transcription_complete(self, output_text):
        """Called when single transcription completes"""
        self.single_output.setText(output_text)
        
        self.transcription_timer.stop()
        self.animation_timer.stop()
        
        elapsed = time.time() - self.transcription_start_time
        self.transcription_status_label.setText("✅ Complete")
        self.transcription_timer_label.setText(f"⏱ {elapsed:.2f}s")
        self.status_label.setText(f"Transcription complete in {elapsed:.2f}s")
    
    def on_continuous_transcription_complete(self, output_text):
        """Called when continuous chunk transcription completes"""
        if output_text.strip():
            self.continuous_output.append(output_text + "\n")
        
        self.transcription_timer.stop()
        self.animation_timer.stop()
        
        elapsed = time.time() - self.transcription_start_time
        self.continuous_status_label.setText("")
        self.continuous_timer_label.setText(f"Last: {elapsed:.2f}s")
    
    def browse_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            self.selected_audio_file = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.transcribe_file_btn.setEnabled(True)
    
    def transcribe_file(self):
        if not hasattr(self, 'selected_audio_file'):
            return
        
        self.file_output.clear()
        self.file_status_label.setText("")
        self.file_timer_label.setText("")
        self.transcribe_file_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)
        
        self.transcription_start_time = time.time()
        self.transcription_timer.start(100)
        self.animation_timer.start()
        self.animation_state = 0
        self.file_status_label.setText("🔄 Transcribing")
        self.status_label.setText("Transcribing file...")
        
        QTimer.singleShot(50, self._do_file_transcription)
    
    def _do_file_transcription(self):
        """Perform the actual file transcription"""
        try:
            segments = self.transcriber.transcribe_file(self.selected_audio_file)
            
            if segments:
                output_text = ""
                for segment in segments:
                    output_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                
                self.file_output.setText(output_text)
                
                elapsed = time.time() - self.transcription_start_time
                self.transcription_timer.stop()
                self.animation_timer.stop()
                
                self.file_status_label.setText("✅ Complete")
                self.file_timer_label.setText(f"⏱ {elapsed:.2f}s")
                self.status_label.setText(f"Transcription complete in {elapsed:.2f}s")
            else:
                self.transcription_timer.stop()
                self.animation_timer.stop()
                self.file_status_label.setText("❌ Failed")
                QMessageBox.warning(self, "Error", "Transcription failed!")
                self.status_label.setText("Transcription failed")
                
        except Exception as e:
            self.transcription_timer.stop()
            self.animation_timer.stop()
            self.file_status_label.setText("❌ Error")
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")
            self.status_label.setText("Error occurred")
        finally:
            self.transcribe_file_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def save_transcription(self):
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:
            text = self.single_output.toPlainText()
        elif current_tab == 1:
            text = self.continuous_output.toPlainText()
        else:
            text = self.file_output.toPlainText()
        
        if not text:
            QMessageBox.warning(self, "No Content", "No transcription to save!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "Success", "Transcription saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def on_error(self, error_msg):
        self.transcription_timer.stop()
        self.animation_timer.stop()
        
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 0:
            self.transcription_status_label.setText("❌ Error")
            self.transcription_timer_label.setText("")
        elif current_tab == 1:
            self.continuous_status_label.setText("❌ Error")
            self.continuous_timer_label.setText("")
        
        QMessageBox.critical(self, "Error", error_msg)
        self.status_label.setText(f"Error: {error_msg}")
        self.progress_bar.setVisible(False)
        
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.continuous_start_btn.setEnabled(True)
        self.continuous_stop_btn.setEnabled(False)
    
    def closeEvent(self, event):
        """Handle application close"""
        self.transcription_timer.stop()
        self.animation_timer.stop()
        
        # Stop thread if running
        if self.recorder_thread is not None:
            try:
                if self.recorder_thread.isRunning():
                    self.recorder_thread.stop()
                    self.recorder_thread.wait(3000)
                    if self.recorder_thread.isRunning():
                        self.recorder_thread.terminate()
            except RuntimeError:
                pass
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    
    # Set fork mode for embedded Linux compatibility
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    
    main()

