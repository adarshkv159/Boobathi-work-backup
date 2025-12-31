# face_recognition_gui.py
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QGroupBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget,
    QFormLayout, QCheckBox, QComboBox, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

from face_detection_scrfd import SCRFD
from face_recognition import Facenet
from face_database import FaceDatabase


class VideoThread(QThread):
    """Thread for video capture and face recognition"""
    frame_ready = Signal(np.ndarray, list, list)
    fps_update = Signal(float)
    error_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.detector = None
        self.facenet = None
        self.face_db = None
        self.cap = None
        self.camera_id = 0
        self.detection_threshold = 0.5
        self.recognition_threshold = 1.15
        
        # ============ SEPARATE NPU CONTROL ============
        self.use_npu_detector = False  # SCRFD on CPU
        self.use_npu_facenet = False   # FaceNet on NPU (configurable)
        self.delegate_path = "/usr/lib/libvx_delegate.so"
        # ==============================================
        
        # ============ DYNAMIC THRESHOLD PARAMETERS ============
        self.face_size_boundary = 120
        self.far_distance_threshold = 1.0
        self.close_distance_threshold = 0.9
        # ======================================================
        
        # Reference points for face alignment
        self.REFERENCE = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
    
    def initialize_models(self, detector_model, facenet_model, use_npu_facenet, delegate_path):
        """Initialize detection and recognition models"""
        try:
            self.use_npu_facenet = use_npu_facenet
            self.delegate_path = delegate_path
            
            print("\n" + "="*70)
            print("MODEL INITIALIZATION")
            print("="*70)
            print(f"Detector model: {detector_model}")
            print(f"FaceNet model: {facenet_model}")
            print(f"SCRFD (detector): CPU (fixed)")
            print(f"FaceNet (recognition): {'NPU' if use_npu_facenet else 'CPU'}")
            print(f"Delegate path: {delegate_path}")
            print(f"Delegate exists: {os.path.exists(delegate_path)}")
            print("="*70)
            
            # Initialize SCRFD detector on CPU (always)
            print("\n[1/2] Initializing SCRFD Detector (CPU)...")
            self.detector = SCRFD(
                detector_model, 
                nms_thresh=0.4, 
                use_npu=False,  # Force CPU for detector
                delegate_path=delegate_path
            )
            print("✓ SCRFD detector initialized on CPU")
            
            # Initialize FaceNet on NPU (if enabled)
            print(f"\n[2/2] Initializing FaceNet ({'NPU' if use_npu_facenet else 'CPU'})...")
            self.facenet = Facenet(
                facenet_model, 
                delegate_path=delegate_path if use_npu_facenet else None
            )
            print(f"✓ FaceNet initialized on {'NPU' if use_npu_facenet else 'CPU'}")
            
            print("\n" + "="*70)
            print("INITIALIZATION COMPLETE")
            print("="*70 + "\n")
            
            self.face_db = FaceDatabase(threshold=self.recognition_threshold)
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR during initialization:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Model initialization failed: {str(e)}")
            return False
    
    def align_face(self, img, kps):
        """Align face using keypoints"""
        kps = np.array(kps, dtype=np.float32)
        transform = cv2.estimateAffinePartial2D(kps, self.REFERENCE, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(img, transform, (112, 112))
        return aligned
    
    def start_capture(self, camera_id):
        """Start video capture"""
        self.camera_id = camera_id
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.error_signal.emit(f"Cannot open camera {camera_id}")
            return False
        
        self.running = True
        self.start()
        return True
    
    def stop_capture(self):
        """Stop video capture"""
        self.running = False
        self.wait()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def update_threshold(self, detection_thresh, recognition_thresh):
        """Update detection and recognition thresholds"""
        self.detection_threshold = detection_thresh
        self.recognition_threshold = recognition_thresh
        if self.face_db:
            self.face_db.threshold = recognition_thresh
    
    def update_dynamic_params(self, boundary, far_thresh, close_thresh):
        """Update dynamic threshold parameters"""
        self.face_size_boundary = boundary
        self.far_distance_threshold = far_thresh
        self.close_distance_threshold = close_thresh
    
    def reload_database(self):
        """Reload face database"""
        if self.face_db:
            self.face_db = FaceDatabase(threshold=self.recognition_threshold)
    
    def run(self):
        """Main video processing loop"""
        import time
        frame_count = 0
        fps_sum = 0
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
            
            start_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                self.error_signal.emit("Failed to read frame")
                break
            
            try:
                # Detect faces (CPU)
                bboxes, kpss = self.detector.detect(frame, thresh=self.detection_threshold)
                
                results = []
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2, score = bbox.astype(int)
                    
                    # Validate coordinates
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    
                    # Extract and align face
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    kp = kpss[i]
                    kp_crop = kp - np.array([x1, y1])
                    aligned_face = self.align_face(face_img, kp_crop)
                    
                    # Get embeddings (NPU if enabled)
                    emb = self.facenet.get_embeddings(aligned_face)
                    
                    # ============ DYNAMIC THRESHOLD BASED ON FACE SIZE ============
                    face_h = y2 - y1
                    face_w = x2 - x1
                    face_size = max(face_h, face_w)
                    
                    if face_size < self.face_size_boundary:
                        dynamic_threshold = self.far_distance_threshold
                        distance_label = "Far"
                    else:
                        dynamic_threshold = self.close_distance_threshold
                        distance_label = "Close"
                    
                    self.face_db.threshold = dynamic_threshold
                    name, conf = self.face_db.find_name(emb)
                    # ==============================================================
                    
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'score': float(score),
                        'name': name,
                        'confidence': conf,
                        'face_size': face_size,
                        'threshold_used': dynamic_threshold,
                        'distance': distance_label
                    })
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_sum += current_fps
                frame_count += 1
                
                if frame_count % 10 == 0:
                    avg_fps = fps_sum / 10
                    self.fps_update.emit(avg_fps)
                    fps_sum = 0
                
                self.frame_ready.emit(frame, results, [])
                
            except Exception as e:
                self.error_signal.emit(f"Processing error: {str(e)}")
        
        if self.cap is not None:
            self.cap.release()


class FaceRecognitionGUI(QMainWindow):
    """Main GUI application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Face Recognition System - PySide6 (FaceNet NPU)")
        self.setGeometry(100, 100, 800, 600)
        
        # Paths and configuration
        self.dataset_dir = "dataset"
        self.detector_model = "model_float32.tflite"
        self.facenet_model = "facenet_512_int_quantized.tflite"
        self.database_file = "database.npy"
        self.use_npu_facenet = False  # FaceNet NPU toggle
        self.delegate_path = "/usr/lib/libvx_delegate.so"
        
        # Create dataset directory
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Video thread
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.fps_update.connect(self.update_fps)
        self.video_thread.error_signal.connect(self.show_error)
        
        # State
        self.current_frame = None
        self.is_running = False
        self.current_fps = 0.0
        
        # Initialize UI first
        self.init_ui()
        
        # Initialize models after UI
        self.initialize_models()
    
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: Video display
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
    
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different sections
        tabs = QTabWidget()
        
        # Dataset management tab
        dataset_tab = self.create_dataset_tab()
        tabs.addTab(dataset_tab, "Dataset")
        
        # Configuration tab
        config_tab = self.create_config_tab()
        tabs.addTab(config_tab, "Configuration")
        
        # Statistics tab
        stats_tab = self.create_stats_tab()
        tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(tabs)
        
        # Control buttons at bottom
        control_group = self.create_control_buttons()
        layout.addWidget(control_group)
        
        return panel
    
    def create_dataset_tab(self):
        """Create dataset management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Person list
        person_group = QGroupBox("Registered Persons")
        person_layout = QVBoxLayout()
        
        self.person_list = QListWidget()
        self.refresh_person_list()
        person_layout.addWidget(self.person_list)
        
        # Person management buttons
        person_btn_layout = QHBoxLayout()
        
        self.btn_add_person = QPushButton("Add Person")
        self.btn_add_person.clicked.connect(self.add_person)
        person_btn_layout.addWidget(self.btn_add_person)
        
        self.btn_remove_person = QPushButton("Remove Person")
        self.btn_remove_person.clicked.connect(self.remove_person)
        person_btn_layout.addWidget(self.btn_remove_person)
        
        person_layout.addLayout(person_btn_layout)
        person_group.setLayout(person_layout)
        layout.addWidget(person_group)
        
        # Image management
        image_group = QGroupBox("Image Management")
        image_layout = QVBoxLayout()
        
        self.btn_add_images = QPushButton("Upload Images")
        self.btn_add_images.clicked.connect(self.add_images)
        image_layout.addWidget(self.btn_add_images)
        
        self.btn_view_images = QPushButton("View Person Images")
        self.btn_view_images.clicked.connect(self.view_person_images)
        image_layout.addWidget(self.btn_view_images)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # Embedding extraction
        embedding_group = QGroupBox("Embedding Extraction")
        embedding_layout = QVBoxLayout()
        
        self.btn_extract_all = QPushButton("Extract All Embeddings")
        self.btn_extract_all.clicked.connect(self.extract_all_embeddings)
        self.btn_extract_all.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        embedding_layout.addWidget(self.btn_extract_all)
        
        self.btn_extract_person = QPushButton("Extract Selected Person")
        self.btn_extract_person.clicked.connect(self.extract_person_embeddings)
        embedding_layout.addWidget(self.btn_extract_person)
        
        self.extraction_status = QLabel("Status: Ready")
        self.extraction_status.setWordWrap(True)
        embedding_layout.addWidget(self.extraction_status)
        
        embedding_group.setLayout(embedding_layout)
        layout.addWidget(embedding_group)
        
        layout.addStretch()
        return widget
    
    def create_config_tab(self):
        """Create configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model paths
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        self.detector_path_input = QLineEdit(self.detector_model)
        model_layout.addRow("Detector Model:", self.detector_path_input)
        
        self.facenet_path_input = QLineEdit(self.facenet_model)
        model_layout.addRow("FaceNet Model:", self.facenet_path_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Detection parameters
        detection_group = QGroupBox("Detection Parameters")
        detection_layout = QFormLayout()
        
        self.detection_threshold = QDoubleSpinBox()
        self.detection_threshold.setRange(0.1, 1.0)
        self.detection_threshold.setSingleStep(0.05)
        self.detection_threshold.setValue(0.5)
        self.detection_threshold.valueChanged.connect(self.update_thresholds)
        detection_layout.addRow("Detection Threshold:", self.detection_threshold)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Dynamic threshold parameters
        dynamic_threshold_group = QGroupBox("Dynamic Recognition Thresholds")
        dynamic_threshold_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        dynamic_layout = QFormLayout()
        
        self.face_size_threshold = QSpinBox()
        self.face_size_threshold.setRange(50, 300)
        self.face_size_threshold.setSingleStep(10)
        self.face_size_threshold.setValue(120)
        self.face_size_threshold.valueChanged.connect(self.update_dynamic_thresholds)
        dynamic_layout.addRow("Face Size Boundary (px):", self.face_size_threshold)
        
        self.far_threshold = QDoubleSpinBox()
        self.far_threshold.setRange(0.5, 2.0)
        self.far_threshold.setSingleStep(0.05)
        self.far_threshold.setValue(1.0)
        self.far_threshold.valueChanged.connect(self.update_dynamic_thresholds)
        dynamic_layout.addRow("Far Distance Threshold:", self.far_threshold)
        
        self.close_threshold = QDoubleSpinBox()
        self.close_threshold.setRange(0.5, 2.0)
        self.close_threshold.setSingleStep(0.05)
        self.close_threshold.setValue(0.9)
        self.close_threshold.valueChanged.connect(self.update_dynamic_thresholds)
        dynamic_layout.addRow("Close Distance Threshold:", self.close_threshold)
        
        info_label = QLabel("• Faces < boundary px use far threshold (relaxed)\n• Faces ≥ boundary px use close threshold (strict)")
        info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic; font-weight: normal;")
        dynamic_layout.addRow("", info_label)
        
        dynamic_threshold_group.setLayout(dynamic_layout)
        layout.addWidget(dynamic_threshold_group)
        
        # ============ HARDWARE ACCELERATION (MODIFIED) ============
        hardware_group = QGroupBox("Hardware Acceleration")
        hardware_layout = QFormLayout()
        
        # Info label
        info_header = QLabel("SCRFD Detector: CPU (fixed)\nFaceNet Recognition: Configurable below")
        info_header.setStyleSheet("color: #2196F3; font-weight: bold; padding: 5px;")
        hardware_layout.addRow("", info_header)
        
        # FaceNet NPU toggle
        self.npu_facenet_checkbox = QCheckBox()
        self.npu_facenet_checkbox.setChecked(self.use_npu_facenet)
        self.npu_facenet_checkbox.stateChanged.connect(self.toggle_npu_facenet)
        hardware_layout.addRow("Use NPU for FaceNet:", self.npu_facenet_checkbox)
        
        self.delegate_path_input = QLineEdit(self.delegate_path)
        hardware_layout.addRow("NPU Delegate Path:", self.delegate_path_input)
        
        hardware_group.setLayout(hardware_layout)
        layout.addWidget(hardware_group)
        # ==========================================================
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()
        
        self.camera_id_input = QSpinBox()
        self.camera_id_input.setRange(0, 10)
        self.camera_id_input.setValue(0)
        camera_layout.addRow("Camera ID:", self.camera_id_input)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Apply button
        self.btn_apply_config = QPushButton("Apply Configuration")
        self.btn_apply_config.clicked.connect(self.apply_configuration)
        self.btn_apply_config.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        layout.addWidget(self.btn_apply_config)
        
        layout.addStretch()
        return widget
    
    def create_stats_tab(self):
        """Create statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance stats
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()
        
        self.fps_label = QLabel("0.0 FPS")
        perf_layout.addRow("Frame Rate:", self.fps_label)
        
        self.backend_label = QLabel("Detector: CPU | FaceNet: CPU")
        perf_layout.addRow("Backend:", self.backend_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Database stats
        db_group = QGroupBox("Database Statistics")
        db_layout = QFormLayout()
        
        self.person_count_label = QLabel("0")
        db_layout.addRow("Registered Persons:", self.person_count_label)
        
        self.embedding_count_label = QLabel("0")
        db_layout.addRow("Total Embeddings:", self.embedding_count_label)
        
        self.btn_refresh_stats = QPushButton("Refresh Statistics")
        self.btn_refresh_stats.clicked.connect(self.refresh_statistics)
        db_layout.addRow("", self.btn_refresh_stats)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Log viewer
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        return widget
    
    def create_control_buttons(self):
        """Create main control buttons"""
        group = QGroupBox("Video Control")
        layout = QVBoxLayout()
        
        self.btn_start = QPushButton("Start Recognition")
        self.btn_start.clicked.connect(self.start_recognition)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("Stop Recognition")
        self.btn_stop.clicked.connect(self.stop_recognition)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 12px;")
        layout.addWidget(self.btn_stop)
        
        self.btn_snapshot = QPushButton("Save Snapshot")
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        self.btn_snapshot.setEnabled(False)
        layout.addWidget(self.btn_snapshot)
        
        group.setLayout(layout)
        return group
    
    def create_right_panel(self):
        """Create right video display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video display
        display_group = QGroupBox("Live Recognition")
        display_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #555;")
        self.video_label.setText("Video feed will appear here")
        
        display_layout.addWidget(self.video_label)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Status: Ready")
        self.info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        self.info_label.setFixedHeight(200)
        info_layout.addWidget(self.info_label)
        layout.addLayout(info_layout)
        
        return panel
    
    def initialize_models(self):
        """Initialize detection and recognition models"""
        success = self.video_thread.initialize_models(
            self.detector_model,
            self.facenet_model,
            self.use_npu_facenet,
            self.delegate_path
        )
        
        if success:
            self.log("Models initialized successfully")
            backend_text = f"Detector: CPU | FaceNet: {'NPU' if self.use_npu_facenet else 'CPU'}"
            self.backend_label.setText(backend_text)
            self.refresh_statistics()
        else:
            self.log("ERROR: Model initialization failed")
    
    def refresh_person_list(self):
        """Refresh the list of registered persons"""
        self.person_list.clear()
        if os.path.exists(self.dataset_dir):
            persons = [d for d in os.listdir(self.dataset_dir) 
                      if os.path.isdir(os.path.join(self.dataset_dir, d))]
            self.person_list.addItems(sorted(persons))
        if hasattr(self, 'person_count_label'):
            self.refresh_statistics()
    
    def add_person(self):
        """Add a new person to the dataset"""
        from PySide6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "Add Person", "Enter person name:")
        if ok and name.strip():
            person_dir = os.path.join(self.dataset_dir, name.strip())
            if os.path.exists(person_dir):
                QMessageBox.warning(self, "Error", f"Person '{name}' already exists!")
            else:
                os.makedirs(person_dir)
                self.refresh_person_list()
                self.log(f"Added person: {name}")
                QMessageBox.information(self, "Success", f"Person '{name}' added successfully!")
    
    def remove_person(self):
        """Remove selected person from dataset"""
        current_item = self.person_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a person to remove!")
            return
        
        person_name = current_item.text()
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete '{person_name}' and all their images?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            person_dir = os.path.join(self.dataset_dir, person_name)
            shutil.rmtree(person_dir)
            
            face_db = FaceDatabase()
            face_db.del_name(person_name)
            
            self.refresh_person_list()
            self.log(f"Removed person: {person_name}")
            QMessageBox.information(self, "Success", f"Person '{person_name}' removed successfully!")
    
    def add_images(self):
        """Add images for selected person"""
        current_item = self.person_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a person first!")
            return
        
        person_name = current_item.text()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images",
            "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if files:
            person_dir = os.path.join(self.dataset_dir, person_name)
            copied = 0
            for file_path in files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(person_dir, filename)
                shutil.copy(file_path, dest_path)
                copied += 1
            
            self.log(f"Added {copied} images for {person_name}")
            QMessageBox.information(self, "Success", f"Added {copied} images for '{person_name}'!")
    
    def view_person_images(self):
        """View images of selected person"""
        current_item = self.person_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a person first!")
            return
        
        person_name = current_item.text()
        person_dir = os.path.join(self.dataset_dir, person_name)
        
        images = [f for f in os.listdir(person_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not images:
            QMessageBox.information(self, "Info", f"No images found for '{person_name}'")
        else:
            msg = f"Person: {person_name}\nTotal images: {len(images)}\n\nImages:\n"
            msg += "\n".join(images[:20])
            if len(images) > 20:
                msg += f"\n... and {len(images) - 20} more"
            QMessageBox.information(self, "Person Images", msg)
    
    def extract_all_embeddings(self):
        """Extract embeddings for all persons in dataset"""
        self.extraction_status.setText("Status: Extracting embeddings...")
        QApplication.processEvents()
        
        try:
            from face_detection_scrfd import SCRFD
            from face_recognition import Facenet
            
            detector = SCRFD(self.detector_model)
            facenet = Facenet(self.facenet_model, delegate_path=None)
            face_db = FaceDatabase(db_file=self.database_file)
            
            face_db.database = {}
            
            REFERENCE = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            
            def align_face(img, kps):
                kps = np.array(kps, dtype=np.float32)
                M = cv2.estimateAffinePartial2D(kps, REFERENCE, method=cv2.LMEDS)[0]
                aligned = cv2.warpAffine(img, M, (112, 112))
                return aligned
            
            total_embeddings = 0
            failed_images = []
            
            for person_name in os.listdir(self.dataset_dir):
                person_folder = os.path.join(self.dataset_dir, person_name)
                if not os.path.isdir(person_folder):
                    continue
                
                person_embeddings = 0
                for file in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, file)
                    if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        continue
                    
                    img = cv2.imread(image_path)
                    if img is None:
                        failed_images.append(image_path)
                        continue
                    
                    bboxes, kpss = detector.detect(img, thresh=0.5)
                    if len(bboxes) == 0:
                        failed_images.append(f"{image_path} (no face detected)")
                        continue
                    
                    kp = kpss[0]
                    x1, y1, x2, y2, _ = bboxes[0].astype(int)
                    crop = img[y1:y2, x1:x2]
                    kp_crop = kp - np.array([x1, y1])
                    aligned = align_face(crop, kp_crop)
                    emb = facenet.get_embeddings(aligned)
                    
                    face_db.add_name(person_name, emb.tolist())
                    person_embeddings += 1
                    total_embeddings += 1
                
                self.log(f"Extracted {person_embeddings} embeddings for {person_name}")
            
            self.video_thread.reload_database()
            
            status_msg = f"Status: Complete! Extracted {total_embeddings} embeddings"
            if failed_images:
                status_msg += f"\n{len(failed_images)} images failed"
            
            self.extraction_status.setText(status_msg)
            self.refresh_statistics()
            
            QMessageBox.information(
                self, "Success",
                f"Extracted {total_embeddings} embeddings!\n\nFailed: {len(failed_images)}"
            )
            
        except Exception as e:
            self.extraction_status.setText(f"Status: Error - {str(e)}")
            QMessageBox.critical(self, "Error", f"Extraction failed:\n{str(e)}")
    
    def extract_person_embeddings(self):
        """Extract embeddings for selected person only"""
        current_item = self.person_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a person first!")
            return
        
        person_name = current_item.text()
        person_folder = os.path.join(self.dataset_dir, person_name)
        
        try:
            from face_detection_scrfd import SCRFD
            from face_recognition import Facenet
            
            detector = SCRFD(self.detector_model)
            facenet = Facenet(self.facenet_model, delegate_path=None)
            face_db = FaceDatabase(db_file=self.database_file)
            
            if person_name in face_db.database:
                face_db.del_name(person_name)
            
            REFERENCE = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            
            def align_face(img, kps):
                kps = np.array(kps, dtype=np.float32)
                M = cv2.estimateAffinePartial2D(kps, REFERENCE, method=cv2.LMEDS)[0]
                aligned = cv2.warpAffine(img, M, (112, 112))
                return aligned
            
            count = 0
            for file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, file)
                if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                bboxes, kpss = detector.detect(img, thresh=0.5)
                if len(bboxes) == 0:
                    continue
                
                kp = kpss[0]
                x1, y1, x2, y2, _ = bboxes[0].astype(int)
                crop = img[y1:y2, x1:x2]
                kp_crop = kp - np.array([x1, y1])
                aligned = align_face(crop, kp_crop)
                emb = facenet.get_embeddings(aligned)
                
                face_db.add_name(person_name, emb.tolist())
                count += 1
            
            self.video_thread.reload_database()
            self.log(f"Extracted {count} embeddings for {person_name}")
            self.refresh_statistics()
            QMessageBox.information(self, "Success", f"Extracted {count} embeddings for '{person_name}'!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Extraction failed:\n{str(e)}")
    
    def update_thresholds(self):
        """Update detection and recognition thresholds"""
        det_thresh = self.detection_threshold.value()
        self.video_thread.detection_threshold = det_thresh
        self.log(f"Updated detection threshold: {det_thresh:.2f}")
    
    def update_dynamic_thresholds(self):
        """Update dynamic threshold parameters"""
        boundary = self.face_size_threshold.value()
        far_thresh = self.far_threshold.value()
        close_thresh = self.close_threshold.value()
        
        self.video_thread.update_dynamic_params(boundary, far_thresh, close_thresh)
        self.log(f"Dynamic thresholds updated: Boundary={boundary}px, Far={far_thresh:.2f}, Close={close_thresh:.2f}")
    
    def toggle_npu_facenet(self, state):
        """Toggle NPU acceleration for FaceNet"""
        self.use_npu_facenet = state == Qt.Checked
        self.log(f"FaceNet NPU: {'Enabled' if self.use_npu_facenet else 'Disabled'}")
    
    def apply_configuration(self):
        """Apply configuration changes"""
        self.detector_model = self.detector_path_input.text()
        self.facenet_model = self.facenet_path_input.text()
        self.delegate_path = self.delegate_path_input.text()
        
        if self.is_running:
            QMessageBox.warning(
                self, "Warning",
                "Please stop recognition before applying configuration!"
            )
            return
        
        self.initialize_models()
        self.log("Configuration applied")
        QMessageBox.information(self, "Success", "Configuration applied successfully!")
    
    def start_recognition(self):
        """Start video recognition"""
        if not os.path.exists(self.database_file):
            reply = QMessageBox.question(
                self, "No Database",
                "No embeddings found. Extract embeddings now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.extract_all_embeddings()
            else:
                return
        
        camera_id = self.camera_id_input.value()
        success = self.video_thread.start_capture(camera_id)
        
        if success:
            self.is_running = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_snapshot.setEnabled(True)
            self.info_label.setText("Status: Running")
            self.log("Started recognition")
        else:
            self.log("ERROR: Failed to start camera")
    
    def stop_recognition(self):
        """Stop video recognition"""
        self.video_thread.stop_capture()
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        self.info_label.setText("Status: Stopped")
        self.video_label.setText("Video feed stopped")
        self.log("Stopped recognition")
    
    def update_frame(self, frame, results, _):
        """Update video frame with detection results"""
        self.current_frame = frame.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            name = result['name']
            conf = result['confidence']
            face_size = result.get('face_size', 0)
            threshold_used = result.get('threshold_used', 0)
            distance = result.get('distance', 'Unknown')
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if name != "Unknown":
                label = f"{name} {conf:.2f} [{distance}]"
            else:
                label = f"Unknown [{distance}]"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), color, -1)
            
            cv2.putText(
                frame, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            info_text = f"Size:{face_size}px T:{threshold_used:.2f}"
            cv2.putText(
                frame, info_text, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        face_count = len(results)
        recognized = sum(1 for r in results if r['name'] != "Unknown")
        self.info_label.setText(
            f"Status: Running | Faces: {face_count} | Recognized: {recognized} | FPS: {self.current_fps:.1f}"
        )
    
    def update_fps(self, fps):
        """Update FPS display"""
        self.current_fps = fps
        self.fps_label.setText(f"{fps:.1f} FPS")
    
    def save_snapshot(self):
        """Save current frame as snapshot"""
        if self.current_frame is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, self.current_frame)
        self.log(f"Saved snapshot: {filename}")
        QMessageBox.information(self, "Success", f"Snapshot saved as {filename}")
    
    def refresh_statistics(self):
        """Refresh database statistics"""
        person_count = len([d for d in os.listdir(self.dataset_dir) 
                           if os.path.isdir(os.path.join(self.dataset_dir, d))])
        self.person_count_label.setText(str(person_count))
        
        if os.path.exists(self.database_file):
            face_db = FaceDatabase(db_file=self.database_file)
            embedding_count = sum(len(embeddings) for embeddings in face_db.database.values())
            self.embedding_count_label.setText(str(embedding_count))
        else:
            self.embedding_count_label.setText("0")
    
    def show_error(self, message):
        """Show error message"""
        self.log(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)
    
    def log(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.is_running:
            self.stop_recognition()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = FaceRecognitionGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

