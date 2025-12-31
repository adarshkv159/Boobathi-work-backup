import sys
import os

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QMessageBox, QScrollArea, QFrame,
    QSizePolicy, QStackedWidget, QSplashScreen
)
from PySide6.QtCore import Qt, QProcess, QTimer, QSize, QRect
from PySide6.QtGui import QFont, QPixmap, QIcon, QColor, QPainter, QLinearGradient, QMovie, QPalette


class LoadingSplash(QSplashScreen):
    """Custom loading splash screen with gradient background and centered logo."""

    def __init__(self, logo_path="Phytec_logo_web_web.png"):
        # Create a pixmap for the splash screen
        pixmap = QPixmap(800, 600)
        super().__init__(pixmap)

        self.logo_path = logo_path
        self.logo_pixmap = None

        # Load logo if it exists
        if os.path.exists(self.logo_path):
            self.logo_pixmap = QPixmap(self.logo_path)

        # Set window flags
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.SplashScreen)

        # Initial draw
        self.draw_splash()

    def draw_splash(self):
        """Draw the gradient background and centered logo."""
        pixmap = QPixmap(800, 600)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create gradient background from #02a6b0 to #2aa952
        gradient = QLinearGradient(0, 0, 800, 600)
        gradient.setColorAt(0.0, QColor("#02a6b0"))
        gradient.setColorAt(1.0, QColor("#2aa952"))

        # Fill background
        painter.fillRect(0, 0, 800, 600, gradient)

        # Draw logo in center if available
        if self.logo_pixmap:
            # Scale logo to reasonable size (max 400px wide, maintain aspect ratio)
            scaled_logo = self.logo_pixmap.scaledToWidth(400, Qt.SmoothTransformation)

            # Calculate center position
            x = (800 - scaled_logo.width()) // 2
            y = (600 - scaled_logo.height()) // 2

            # Draw logo
            painter.drawPixmap(x, y, scaled_logo)
        else:
            # Fallback text if logo not found
            painter.setPen(QColor("#ffffff"))
            font = QFont("Arial", 48, QFont.Bold)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "PHYTEC")

        # Draw loading text
        painter.setPen(QColor("#ffffff"))
        font = QFont("Segoe UI", 16, QFont.Bold)
        painter.setFont(font)
        text_rect = QRect(0, 520, 800, 40)
        painter.drawText(text_rect, Qt.AlignCenter, "Loading AI/ML Demo Launcher...")

        painter.end()

        # Set the pixmap
        self.setPixmap(pixmap)


class FlowGrid(QWidget):
    """A simple flow-like grid that reflows child cards on resize for responsive columns."""

    def __init__(self, hgap=10, vgap=10, min_col_width=240, parent=None):
        super().__init__(parent)
        self._hgap = hgap
        self._vgap = vgap
        self._min_col_width = min_col_width
        self.fixed_columns = 3

    def setMinColWidth(self, w):
        self._min_col_width = w
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        return QSize(600, 400)

    def minimumSizeHint(self):
        return QSize(200, 200)

    def _cardSize(self, child):
        hint = child.sizeHint()
        w = max(self._min_col_width, hint.width())
        h = max(130, hint.height())
        return QSize(w, h)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._doLayout()

    def _doLayout(self):
        x = self._hgap
        y = self._vgap
        line_height = 0
        avail_w = max(100, self.width())

        for child in self.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            if not child.isVisible():
                continue

            if self.fixed_columns:
                sz = self._cardSize(child)
                col_width = (avail_w - (self._hgap * (self.fixed_columns + 1))) // self.fixed_columns
                sz.setWidth(col_width)
            else:
                sz = self._cardSize(child)

            # Check if card fits on current line
            if x + sz.width() > avail_w and x > self._hgap:
                # Move to next line
                x = self._hgap
                y += line_height + self._vgap
                line_height = 0

            # Place the card
            child.setGeometry(QRect(x, y, sz.width(), sz.height()))

            # Update position for next card
            x += sz.width() + self._hgap
            line_height = max(line_height, sz.height())

        # Adjust container height
        total_h = y + line_height + self._vgap
        self.setMinimumHeight(total_h)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._doLayout)


class CoverImage(QLabel):
    """Label that displays a center-cropped image without distortion."""

    def __init__(self, img_path=None, height=140):
        super().__init__()
        self.setFixedHeight(height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #222; border: none;")
        self.pix = None

        if img_path and os.path.exists(img_path):
            self.pix = QPixmap(img_path)

    def paintEvent(self, event):
        if not self.pix:
            return super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # destination rectangle
        target = self.rect()

        # compute scaled pixmap while keeping aspect ratio
        scaled = self.pix.scaled(
            target.width(),
            target.height(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )

        # center crop
        x = (scaled.width() - target.width()) // 2
        y = (scaled.height() - target.height()) // 2

        painter.drawPixmap(target, scaled, QRect(x, y, target.width(), target.height()))
        painter.end()


class DemoCard(QFrame):
    """Clickable demo card with full-width cover image."""

    def __init__(self, demo, on_click):
        super().__init__()
        self.demo = demo
        self.on_click = on_click
        self.setObjectName("demoCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Main layout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 1) COVER IMAGE AREA
        img_path = demo.get("image", None)
        self.cover = CoverImage(img_path, height=180)
        lay.addWidget(self.cover)

        img_path = demo.get("image", None)
        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path).scaled(
                400, 140,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            self.cover.setPixmap(pix)
        else:
            # fallback if missing
            self.cover.setStyleSheet("background:#2b2b2b; color:white;")
            self.cover.setText("No Image")
            self.cover.setAlignment(Qt.AlignCenter)

        lay.addWidget(self.cover)

        # 2) CONTENT SECTION
        body = QVBoxLayout()
        body.setContentsMargins(12, 10, 12, 12)
        body.setSpacing(6)

        # Title
        title = QLabel(demo["name"])
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        body.addWidget(title)

        # Badges row
        badges = QHBoxLayout()
        badges.setSpacing(6)

        cat = QLabel(demo["category"])
        cat.setProperty("badge", "blue")
        badges.addWidget(cat)

        mdl = QLabel(demo["models"])
        mdl.setProperty("badge", "green")
        badges.addWidget(mdl)

        badges.addStretch()
        body.addLayout(badges)

        # Description
        desc = QLabel(demo["description"])
        desc.setObjectName("cardDesc")
        desc.setWordWrap(True)

        fm = desc.fontMetrics()
        clipped = fm.elidedText(demo["description"], Qt.ElideRight, 350)
        desc.setText(clipped)

        body.addWidget(desc)
        lay.addLayout(body)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if callable(self.on_click):
                self.on_click(self.demo)
        super().mouseReleaseEvent(e)


class DemoLauncher(QMainWindow):
    """Responsive grid-first AI/ML Demo Launcher with full-screen detail page."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PHYTEC AI/ML Demo Launcher - phyBOARD-pollux i.MX8M Plus")
        self.resize(1400, 850)
        self.setMinimumSize(1000, 650)

        self.running_processes = {}
        self._logo_pixmap = None
        self.current_filter = "All"
        self._current_demo = None

        # Demo config
        self.demos = [
            {
                "id": 1,
                "name": "Object Detection",
                "description": "Real-time object detection using SSD MobileNet with bounding boxes and confidences.",
                "path": "01-object_detection/qt-object-app-2.py",
                "icon": "",
                "category": "Computer Vision",
                "models": "SSD MobileNet V1",
                "features": [
                    "Real-time detection",
                    "Multi-object tracking",
                    "CPU/NPU support"
                ],
                "image": "demo-cover-images/demo-1.jpg",
            },
            {
                "id": 2,
                "name": "Image Classification",
                "description": "Classify images into 1000 categories using MobileNet V2 with top-5 predictions.",
                "path": "02-image_classification/qt-image-classification.py",
                "icon": "",
                "category": "Computer Vision",
                "models": "MobileNet V2",
                "features": [
                    "1000 classes",
                    "Top-5 predictions",
                    "Batch processing",
                ],
                "image": "demo-cover-images/demo-2.png",
            },
            {
                "id": 3,
                "name": "Selfie Segmentation",
                "description": "Background removal and person segmentation in real time for portraits.",
                "path": "03-selfie-segmenter/qt-selfie-segmenter.py",
                "icon": "",
                "category": "Computer Vision",
                "models": "MediaPipe Selfie",
                "features": [
                    "Real-time segmentation",
                    "Background blur",
                    "Virtual backgrounds",
                ],
                "image": "demo-cover-images/demo-3.png",
            },
            {
                "id": 4,
                "name": "Pneumonia Detection",
                "description": "Pneumonia detection from chest X-rays with confidence scoring.",
                "path": "04-pneumonia/qt-pneumonia-detection.py",
                "icon": "",
                "category": "Medical AI",
                "models": "Custom CNN",
                "features": [
                    "X-ray analysis",
                    "Binary classification",
                    "Medical imaging",
                ],
                "image": "demo-cover-images/demo-4.png",
            },
            {
                "id": 5,
                "name": "Number Plate Detection",
                "description": "ALPR: detect and read vehicle license plates.",
                "path": "05-numberplate_extraction/qt-number-plate-detection.py",
                "icon": "",
                "category": "Computer Vision",
                "models": "YOLO + OCR",
                "features": [
                    "Plate detection",
                    "Character recognition",
                    "Vehicle tracking",
                ],
                "image": "demo-cover-images/demo-5.png",
            },
            {
                "id": 6,
                "name": "Pose Estimation",
                "description": "17-point skeleton tracking for motion analysis.",
                "path": "06-pose_detection/qt-pose-estimation.py",
                "icon": "",
                "category": "Computer Vision",
                "models": "ResNet50 Pose",
                "features": [
                    "17 keypoints",
                    "Skeleton visualization",
                    "Multi-person tracking",
                ],
                "image": "demo-cover-images/demo-6.png",
            },
            {
                "id": 7,
                "name": "Face Recognition",
                "description": "Face recognition with voice commands for managing identities.",
                "path": "07-face_recognition/qt-face-full-application.py",
                "icon": "",
                "category": "Computer Vision + NLP",
                "models": "FaceNet + Whisper",
                "features": [
                    "Face recognition",
                    "Voice commands",
                    "Database management",
                ],
                "image": "demo-cover-images/demo-7.png",
            },
            {
                "id": 8,
                "name": "Hand Gesture Detection",
                "description": "21-point hand tracking and gesture recognition.",
                "path": "08-gesture_detection/qt-hand-gesture-detection.py",
                "icon": "",
                "category": "Computer Vision",
                "models": "MediaPipe Hands",
                "features": [
                    "21 landmarks",
                    "Gesture recognition",
                    "Hand skeleton",
                ],
                "image": "demo-cover-images/demo-8.png",
            },
            {
                "id": 9,
                "name": "Driver Monitoring System",
                "description": "Drowsiness, yawn, and distraction alerts for drivers.",
                "path": "09-driver_monitoring_system/qt-DMS-system.py",
                "icon": "",
                "category": "Automotive AI",
                "models": "Face + Eye Landmarks",
                "features": [
                    "Drowsiness detection",
                    "Yawn detection",
                    "Distraction alerts",
                ],
                "image": "demo-cover-images/demo-9.png",
            },
            {
                "id": 10,
                "name": "Lane Detection",
                "description": "Real-time lane detection for ADAS and autonomy.",
                "path": "10-lane_detection/qt-lane-detection.py",
                "icon": "",
                "category": "Automotive AI",
                "models": "UltraFast Lane",
                "features": [
                    "Multi-lane detection",
                    "Real-time tracking",
                    "ADAS support",
                ],
                "image": "demo-cover-images/demo-10.png",
            },
            {
                "id": 11,
                "name": "Brain Tumor detection",
                "description": "Brain Tumor detection on MRI images.",
                "path": "11-Brain-tumor-detection/app-on-npu.py",
                "icon": "",
                "category": "Medical AI",
                "models": "VGG16 classification",
                "features": [
                    "Multi-type Brain Tumor detection",
                    "MRI Images classification",
                ],
                "image": "demo-cover-images/demo-11.jpg",
            },
        ]

        self._build_ui()
        self._set_theme()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)

        root = QVBoxLayout(cw)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        header = self._create_header()
        header.setSizePolicy(header.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
        root.addWidget(header)

        # Central stacked area: page 0 = grid, page 1 = details
        self.stack = QStackedWidget()
        self.grid_page = self._create_grid_page()
        self.details_page = self._create_details_page()
        self.stack.addWidget(self.grid_page)   # index 0
        self.stack.addWidget(self.details_page)  # index 1
        root.addWidget(self.stack, 1)

        footer = self._create_footer()
        footer.setSizePolicy(footer.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
        root.addWidget(footer)

        self._populate_grid()

    def _create_header(self):
        header = QFrame()
        header.setProperty("role", "header")
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 12, 20, 12)
        h.setSpacing(16)

        logo_box = QWidget()
        vb = QVBoxLayout(logo_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setAlignment(Qt.AlignVCenter)

        self.logo_label = QLabel()
        self.logo_label.setMinimumSize(10, 10)
        self.logo_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        if os.path.exists("Phytec_logo_web_web.png"):
            self._logo_pixmap = QPixmap("Phytec_logo_web_web.png")
            self._apply_logo_pixmap()
        elif os.path.exists("Phytec_logo_web_web.jpg"):
            self._logo_pixmap = QPixmap("Phytec_logo_web_web.jpg")
            self._apply_logo_pixmap()
        else:
            self.logo_label.setText("PHYTEC")
            self.logo_label.setFont(QFont("Arial", 28, QFont.Bold))
            self.logo_label.setStyleSheet("QLabel { color: #4a9eff; }")

        vb.addWidget(self.logo_label)
        h.addWidget(logo_box, 0)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(2)
        sep.setObjectName("accentLine")
        h.addWidget(sep)

        title_box = QWidget()
        tv = QVBoxLayout(title_box)
        tv.setContentsMargins(8, 0, 0, 0)
        tv.setSpacing(2)

        self.title_label = QLabel("AI/ML Demo Launcher")
        self.title_label.setObjectName("hdrTitle")
        tv.addWidget(self.title_label)

        self.subtitle_label = QLabel("phyBOARD-pollux i.MX8M Plus Platform")
        self.subtitle_label.setObjectName("hdrSubtitle")
        tv.addWidget(self.subtitle_label)

        self.platform_info_label = QLabel("NXP i.MX8M Plus SoC • Neural Processing Unit Accelerated")
        self.platform_info_label.setObjectName("hdrInfo")
        tv.addWidget(self.platform_info_label)

        h.addWidget(title_box, 1)

        stats_container = QWidget()
        stats_container.setObjectName("statsBadge")
        sv = QVBoxLayout(stats_container)
        sv.setAlignment(Qt.AlignCenter)

        self.stats_number = QLabel(str(len(self.demos)))
        self.stats_number.setObjectName("statsNumber")
        self.stats_text = QLabel("Demos")
        self.stats_text.setObjectName("statsText")

        sv.addWidget(self.stats_number)
        sv.addWidget(self.stats_text)
        stats_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        h.addWidget(stats_container, 0)

        return header

    def _apply_logo_pixmap(self):
        if not self._logo_pixmap:
            return
        target_h = max(40, min(72, int(self.height() * 0.06)))
        pr = self.devicePixelRatioF()
        scaled = self._logo_pixmap.scaledToHeight(int(target_h * pr), Qt.SmoothTransformation)
        scaled.setDevicePixelRatio(pr)
        self.logo_label.setPixmap(scaled)
        self.logo_label.setFixedSize(int(scaled.width() / pr), int(scaled.height() / pr))

    # ============ Page 0: Grid ============

    def _create_grid_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(8)

        grid_group = QGroupBox("Available Demonstrations")
        vg = QVBoxLayout(grid_group)
        vg.setContentsMargins(12, 12, 12, 12)
        vg.setSpacing(8)

        filter_label = QLabel("FILTER BY CATEGORY")
        filter_label.setObjectName("caption")
        vg.addWidget(filter_label)

        filters = QHBoxLayout()
        filters.setSpacing(6)

        categories = [
            ("All", "All"),
            ("Computer Vision", "Computer Vision"),
            ("Auto", "Automotive AI"),
            ("Medical", "Medical AI"),
            ("NLP", "Computer Vision + NLP"),
        ]

        for btn_text, category in categories:
            btn = QPushButton(f"{btn_text}")
            btn.clicked.connect(lambda checked=False, c=category: self._filter_demos(c))
            btn.setProperty("tagButton", True)
            btn.setCursor(Qt.PointingHandCursor)
            filters.addWidget(btn)

        vg.addLayout(filters)

        # Scrollable flow grid
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QFrame.NoFrame)

        self.grid_container = FlowGrid(hgap=10, vgap=10, min_col_width=260)
        self.grid_container.setObjectName("gridContainer")
        self.grid_scroll.setWidget(self.grid_container)

        vg.addWidget(self.grid_scroll, 1)

        self.grid_count_label = QLabel()
        self.grid_count_label.setObjectName("metaText")
        vg.addWidget(self.grid_count_label)

        v.addWidget(grid_group, 1)
        return page

    def _populate_grid(self, filter_category=None):
        # Hide grid during rebuild to avoid overlap flash
        self.grid_container.setVisible(False)

        # Clear previous cards
        for child in self.grid_container.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            child.setParent(None)
            child.deleteLater()

        count = 0
        for demo in self.demos:
            if filter_category and filter_category != "All" and demo["category"] != filter_category:
                continue
            card = DemoCard(demo, self._on_card_selected)
            card.setParent(self.grid_container)
            card.show()
            count += 1

        self.stats_number.setText(str(count))
        if filter_category and filter_category != "All":
            self.grid_count_label.setText(f"Showing {count} of {len(self.demos)} demos • Filter: {filter_category}")
        else:
            self.grid_count_label.setText(f"Total: {count} demonstrations available")

        # Force layout recalculation
        QTimer.singleShot(0, self.grid_container._doLayout)
        self.grid_container.setVisible(True)
        self.grid_container.update()

    def _filter_demos(self, category):
        self.current_filter = category
        self._populate_grid(category if category != "All" else None)
        # Always return to grid page when changing filters
        if self.stack.currentIndex() != 0:
            self.stack.setCurrentIndex(0)

    # ============ Page 1: Details ============

    def _create_details_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(10)

        # Top bar with Back
        top = QHBoxLayout()
        self.back_button = QPushButton("Back to Demos")
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.clicked.connect(self._go_back_to_grid)
        self.back_button.setProperty("action", "primary")
        self.back_button.setMinimumHeight(36)
        top.addWidget(self.back_button, 0)
        top.addStretch(1)
        v.addLayout(top)

        panel = QGroupBox("Demo Information and Controls")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 8, 8, 8)
        pv.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_widget.setMaximumWidth(600)
        details_layout.setSpacing(12)
        details_layout.setContentsMargins(12, 12, 12, 12)

        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setAlignment(Qt.AlignHCenter)
        wrapper_layout.addWidget(details_widget)
        scroll.setWidget(wrapper)

        self.detail_title = QLabel("Select a demonstration from the grid")
        self.detail_title.setObjectName("detailTitle")
        self.detail_title.setWordWrap(True)
        details_layout.addWidget(self.detail_title)

        badge_layout = QHBoxLayout()
        badge_layout.setSpacing(8)

        self.detail_category = QLabel("Category: -")
        self.detail_category.setProperty("badge", "blue")
        badge_layout.addWidget(self.detail_category)

        self.detail_models = QLabel("Model: -")
        self.detail_models.setProperty("badge", "green")
        badge_layout.addWidget(self.detail_models)

        badge_layout.addStretch()
        details_layout.addLayout(badge_layout)

        desc_header = QLabel("DESCRIPTION")
        desc_header.setObjectName("caption")
        details_layout.addWidget(desc_header)

        self.detail_description = QLabel("No demo selected")
        self.detail_description.setWordWrap(True)
        self.detail_description.setObjectName("monoPanel")
        details_layout.addWidget(self.detail_description)

        features_header = QLabel("KEY FEATURES")
        features_header.setObjectName("caption")
        details_layout.addWidget(features_header)

        self.detail_features = QTextEdit()
        self.detail_features.setReadOnly(True)
        self.detail_features.setMaximumHeight(120)
        self.detail_features.setObjectName("monoEdit")
        details_layout.addWidget(self.detail_features)

        # Status label (used instead of old unused function)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setObjectName("statusInfo")
        details_layout.addWidget(self.status_label)

        details_layout.addStretch()

        controls_container = QFrame()
        controls_container.setObjectName("controlsPane")
        controls_container.setMaximumWidth(240)
        c = QVBoxLayout(controls_container)
        c.setContentsMargins(6, 6, 6, 6)
        c.setSpacing(6)

        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        button_layout.setSpacing(8)

        self.launch_button = QPushButton("LAUNCH DEMO")
        self.launch_button.setEnabled(False)
        self.launch_button.clicked.connect(self.launch_demo)
        self.launch_button.setCursor(Qt.PointingHandCursor)
        self.launch_button.setMinimumHeight(30)
        self.launch_button.setMinimumWidth(100)
        self.launch_button.setProperty("action", "primary")
        button_layout.addWidget(self.launch_button)

        button_layout.addSpacing(12)

        self.stop_button = QPushButton("STOP DEMO")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_demo)
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setMinimumHeight(30)
        self.stop_button.setMinimumWidth(100)
        self.stop_button.setProperty("action", "danger")
        button_layout.addWidget(self.stop_button)

        c.addLayout(button_layout)

        # TWO COLUMN LAYOUT
        content_row = QHBoxLayout()
        content_row.setSpacing(20)

        # left = details scroll area
        content_row.addWidget(scroll, 2)

        # right = controls panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        
        # === LABEL ABOVE PREVIEW ===
        preview_header = QLabel("Demo Output")
        preview_header.setObjectName("caption")   # matches your theme style
        preview_header.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(preview_header)

        # PREVIEW GIF / IMAGE
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(350, 260)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            """
            background-color: #0f1621;
            border: 1px solid #2c3e50;
            border-radius: 8px;
            """
        )
        right_panel.addWidget(self.preview_label, alignment=Qt.AlignTop)

        # BUTTONS
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignCenter)

        self.launch_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_row.addWidget(self.launch_button)
        btn_row.addWidget(self.stop_button)

        right_panel.addLayout(btn_row)

        content_row.addLayout(right_panel, 1)
        pv.addLayout(content_row)

        v.addWidget(panel, 1)
        return page

    def _go_back_to_grid(self):
        self.stack.setCurrentIndex(0)

    def _on_card_selected(self, demo):
        gif_path = f"demo-gifs/demo-{demo['id']}.gif"
        if os.path.exists(gif_path):
            movie = QMovie(gif_path)
            movie.setScaledSize(self.preview_label.size())
            self.preview_label.setMovie(movie)
            movie.start()
        else:
            self.preview_label.setText("No preview available")

        self._current_demo = demo
        self.detail_title.setText(f"{demo['name']}")
        self.detail_category.setText(f"Category: {demo['category']}")
        self.detail_models.setText(f"Model: {demo['models']}")
        self.detail_description.setText(demo["description"])

        features_text = "\n".join(f"- {f}" for f in demo["features"])
        self.detail_features.setText(features_text)

        exists = os.path.exists(demo["path"])
        self.launch_button.setEnabled(exists and (demo["id"] not in self.running_processes))
        self.stop_button.setEnabled(demo["id"] in self.running_processes)

        self.stack.setCurrentIndex(1)

    # Simple status setter replacing unused function pattern
    def _set_status(self, text, cls):
        self.status_label.setText(text)
        self.status_label.setObjectName(cls)
        # Force style refresh
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    # Process control
    def launch_demo(self):
        demo = getattr(self, "_current_demo", None)
        if not demo:
            return

        if demo["id"] in self.running_processes:
            QMessageBox.warning(self, "Already Running", f"{demo['name']} is already running!")
            return

        if not os.path.exists(demo["path"]):
            QMessageBox.critical(self, "File Not Found", f"Demo file not found:\n{demo['path']}")
            return

        abs_demo_path = os.path.abspath(demo["path"])
        demo_dir = os.path.dirname(abs_demo_path)
        demo_file = os.path.basename(abs_demo_path)

        process = QProcess(self)
        process.setWorkingDirectory(demo_dir)
        process.setProgram("python3")
        process.setArguments([demo_file])

        process.started.connect(lambda: self._on_demo_started(demo))
        process.finished.connect(
            lambda exit_code, exit_status: self._on_demo_finished(demo, exit_code, exit_status)
        )
        process.errorOccurred.connect(lambda error: self._on_demo_error(demo, error))
        process.readyReadStandardOutput.connect(lambda: self._handle_stdout(demo["id"]))
        process.readyReadStandardError.connect(lambda: self._handle_stderr(demo["id"]))

        self.running_processes[demo["id"]] = process
        self._set_status(f"Status: {demo['name']} Launching…", "statusInfo")
        self.stop_button.setEnabled(True)
        self.launch_button.setEnabled(False)
        process.start()

    def stop_demo(self):
        demo = getattr(self, "_current_demo", None)
        if not demo:
            return
        if demo["id"] not in self.running_processes:
            return
        process = self.running_processes[demo["id"]]
        process.terminate()
        QTimer.singleShot(2000, lambda: process.kill() if process.state() != QProcess.NotRunning else None)

    def _handle_stdout(self, demo_id):
        if demo_id in self.running_processes:
            p = self.running_processes[demo_id]
            out = p.readAllStandardOutput().data().decode()
            if out.strip():
                print(f"[Demo {demo_id}] {out.strip()}")

    def _handle_stderr(self, demo_id):
        if demo_id in self.running_processes:
            p = self.running_processes[demo_id]
            out = p.readAllStandardError().data().decode()
            if out.strip():
                print(f"[Demo {demo_id} ERROR] {out.strip()}")

    def _on_demo_started(self, demo):
        self._set_status(f"Status: {demo['name']} Running", "statusRun")
        self.stop_button.setEnabled(True)
        self.launch_button.setEnabled(False)

    def _on_demo_finished(self, demo, exit_code, exit_status):
        if demo["id"] in self.running_processes:
            del self.running_processes[demo["id"]]

        if exit_code == 0:
            self._set_status(f"Status: {demo['name']} Completed Successfully", "statusOk")
        else:
            self._set_status(f"Status: {demo['name']} Exited (Code {exit_code})", "statusWarn")

        self.stop_button.setEnabled(False)
        self.launch_button.setEnabled(True)

    def _on_demo_error(self, demo, error):
        if demo["id"] in self.running_processes:
            del self.running_processes[demo["id"]]
        self._set_status("Status: Error", "statusErr")
        self.stop_button.setEnabled(False)
        self.launch_button.setEnabled(True)

    def _create_footer(self):
        footer = QFrame()
        footer.setProperty("role", "footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 6, 20, 6)
        layout.setSpacing(10)

        info_label = QLabel("Select a demo card from the grid to view details and launch.")
        info_label.setObjectName("footerInfo")
        layout.addWidget(info_label)

        layout.addStretch()

        copyright_label = QLabel("© 2025 PHYTEC | NXP i.MX8M Plus | Qt6 Framework")
        copyright_label.setObjectName("footerCopy")
        layout.addWidget(copyright_label)

        return footer

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        w = self.width()
        self.title_label.setFont(QFont("Segoe UI", max(20, min(32, w // 60)), QFont.Bold))
        self.subtitle_label.setFont(QFont("Segoe UI", max(10, min(16, w // 100))))
        self.platform_info_label.setFont(QFont("Segoe UI", max(9, min(13, w // 110))))
        self._apply_logo_pixmap()
        self.grid_container.setMinColWidth(350)

        # Relayout grid when resizing to avoid temporary overlap
        if self.stack.currentIndex() == 0:
            self.grid_container._doLayout()

    def _set_theme(self):
        # Ensure the main window base is pure black
        self.setStyleSheet(
            """
QMainWindow { background-color: #000000; }
QWidget { color: #ecf0f1; font-family: 'Segoe UI', 'Arial', sans-serif; }

/* Accent palette */
/* Primary A: #02a6b1, Primary B: #29a955 */
/* Derivatives for borders/shadows */
/* Darken A: #028a93, Darken B: #238a46 */

QGroupBox {
    font-size: 16px; font-weight: 600; color: #ffffff;
    border: 1px solid #2d425f; border-radius: 10px; margin-top: 14px; padding-top: 18px;
    background-color: #121a26;
}
QGroupBox::title { left: 14px; padding: 2px 8px; background-color: transparent; }

/* Header and footer with new gradient */
QFrame[role="header"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #02a6b1, stop:1 #29a955);
    border-bottom: 2px solid #02a6b1;
}
QFrame[role="footer"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #02a6b1, stop:1 #29a955);
    border-top: 2px solid #02a6b1;
}

#accentLine { background-color: #02a6b1; }
#hdrTitle { color: #ffffff; font-size: 28px; font-weight: 800; }
#hdrSubtitle { color: #cfeeee; font-size: 13px; }
#hdrInfo { color: #a6e2c0; font-size: 12px; }

QWidget#statsBadge {
    background-color: rgba(2,166,177,0.22); border: 1px solid #02a6b1;
    border-radius: 12px; padding: 10px;
}
#statsNumber { color: #ffffff; font-size: 30px; font-weight: 800; }
#statsText { color: #cfeeee; font-size: 12px; font-weight: 700; }

/* Badges */
QLabel[badge="blue"] {
    padding: 6px 10px; background-color: rgba(2,166,177,0.18);
    border-radius: 6px; color: #02a6b1; font-size: 11px; font-weight: 700;
    border: 1px solid #02a6b1;
}
QLabel[badge="green"] {
    padding: 6px 10px; background-color: rgba(41,169,85,0.18);
    border-radius: 6px; color: #29a955; font-size: 11px; font-weight: 700;
    border: 1px solid #29a955;
}

/* Detail title panel */
QLabel#detailTitle {
    color: #ffffff; padding: 14px;
    background: #243447; border-radius: 8px;
    border-left: 5px solid #02a6b1;
    font-size: 20px; font-weight: 800;
}
QLabel#monoPanel {
    padding: 12px; background-color: #0f1621; border-radius: 8px; color: #ecf0f1;
    line-height: 1.6; font-size: 14px; border: 1px solid #2c3e50;
}
QTextEdit#monoEdit {
    background-color: #0f1621; border: 1px solid #2c3e50; border-radius: 8px; color: #9fe8b5;
    padding: 12px; font-size: 14px; line-height: 1.5;
}
QLabel#pathPanel {
    padding: 10px; background-color: #0f1621; border-radius: 7px; color: #95a5a6;
    font-family: 'Consolas','Courier New',monospace; font-size: 12px; border: 1px solid #2c3e50;
}

/* Status chips */
QLabel#statusOk , QLabel#statusRun, QLabel#statusWarn, QLabel#statusErr, QLabel#statusInfo {
    padding: 4px; color: white; border-radius: 6px; font-weight: 800; font-size: 11px;
    border: 1px solid transparent;
}
QLabel#statusOk { background: #29a955; border-color: #238a46; }
QLabel#statusRun { background: #02a6b1; border-color: #028a93; }
QLabel#statusWarn { background: #e67e22; border-color: #d68910; }
QLabel#statusErr { background: #c0392b; border-color: #a93226; }
QLabel#statusInfo {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #02a6b1, stop:1 #29a955);
    border-color: #028a93;
}

QFrame#controlsPane {
    background-color: #131b27; border: 1px solid #2c3e50; border-radius: 8px; padding: 10px;
}

/* Buttons */
QPushButton { border-radius: 8px; padding: 6px 10px; font-size: 14px; font-weight: 700; color: white; }
QPushButton:hover { filter: brightness(1.08); }
QPushButton:disabled { background-color: #2c3e50; color: #7f8c8d; border: 1px solid #34495e; }
QPushButton[action="primary"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #02a6b1, stop:1 #29a955);
    border: 1px solid #028a93;
}
QPushButton[action="primary"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #03b6c2, stop:1 #2fbc5f);
    border-color: #03a0ab;
}
QPushButton[action="danger"] { background: #e74c3c; color: white; border: 1px solid #c0392b; }

/* Tag buttons (filters) */
QPushButton[tagButton="true"] {
    background-color: #1a2332; color: #ecf0f1; border: 1px solid #2c3e50;
    border-radius: 6px; padding: 8px 10px; font-size: 11px; font-weight: 700;
}
QPushButton[tagButton="true"]:hover {
    border-color: #02a6b1; color: white;
    box-shadow: 0px 0px 0px 2px rgba(2,166,177,0.12);
}

/* Grid cards */
#gridContainer { background: transparent; }
QFrame#demoCard {
    background-color: #1a2332; border: 1px solid #2c3e50; border-radius: 10px;
}
QFrame#demoCard:hover { border-color: #02a6b1; box-shadow: 0px 0px 0px 2px rgba(2,166,177,0.15); }
QLabel#cardTitle { font-size: 15px; font-weight: 800; color: #ffffff; }
QLabel#cardDesc { font-size: 12px; color: #b0bec5; }

/* Scrollbars with accent */
QScrollBar:vertical {
    background-color: #1a2332; width: 12px; border-radius: 6px; margin: 2px;
}
QScrollBar::handle:vertical {
    background: #02a6b1; border-radius: 6px; min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

QScrollBar:horizontal {
    background-color: #1a2332; height: 12px; border-radius: 6px; margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #29a955; border-radius: 6px; min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

QLabel#caption { color: #a8e7db; font-size: 11px; font-weight: 800; letter-spacing: 1px; }
QFrame#thinLine { background-color: #2f4a6b; max-height: 1px; }
QLabel#footerInfo { color: #cfeeee; font-size: 13px; }
QLabel#footerCopy { color: #a6e2c0; font-size: 12px; font-weight: 500; }



"""
        )

    def closeEvent(self, event):
        if self.running_processes:
            reply = QMessageBox.question(
                self,
                "Running Demos",
                f"{len(self.running_processes)} demonstration(s) currently running.\n\nTerminate all and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                for process in list(self.running_processes.values()):
                    process.terminate()
                    process.waitForFinished(2000)
                    if process.state() != QProcess.NotRunning:
                        process.kill()
                self.running_processes.clear()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Global dark palette
    pal = app.palette()
    pal.setColor(QPalette.Window, Qt.black)
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, Qt.black)
    pal.setColor(QPalette.AlternateBase, Qt.black)
    pal.setColor(QPalette.ToolTipBase, Qt.black)
    pal.setColor(QPalette.ToolTipText, Qt.white)
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, Qt.black)
    pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Highlight, QColor("#02a6b1"))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)

    if os.path.exists("phytec-logo-p.png"):
        app.setWindowIcon(QIcon("phytec-logo-p.png"))

    # Loading splash screen
    splash = LoadingSplash("Phytec_logo_web_web.png")
    splash.show()
    app.processEvents()

    QTimer.singleShot(2000, lambda: None)
    app.processEvents()

    launcher = DemoLauncher()
    splash.finish(launcher)
    launcher.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
