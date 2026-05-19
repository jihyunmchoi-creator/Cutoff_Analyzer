import sys
import cv2
import numpy as np
import math
import os

# PyQt6 및 Matplotlib 설정
os.environ["QT_API"] = "pyqt6"
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QSizePolicy, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# 다크 모드 스타일 시트
DARK_STYLE = """
    QMainWindow { background-color: #121212; }
    QWidget { background-color: #121212; color: #e0e0e0; }
    QLabel { color: #e0e0e0; font-weight: bold; }
    QLineEdit { background-color: #2c2c2c; color: #ffffff; border: 1px solid #444; border-radius: 4px; padding: 3px; }
    QComboBox { background-color: #2c2c2c; color: #ffffff; border: 1px solid #444; border-radius: 4px; }
    QPushButton { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555; border-radius: 6px; padding: 5px 15px; }
    QPushButton:hover { background-color: #4d4d4d; }
    QPushButton#ArrowBtn { padding: 0px; font-size: 8px; min-width: 20px; max-width: 20px; min-height: 12px; max-height: 12px; border-radius: 2px; }
    QPushButton#RunBtn { background-color: #0a84ff; border: none; font-weight: bold; font-size: 18px; }
    QPushButton#RunBtn:hover { background-color: #409fff; }
    QPushButton#SubResetBtn { background-color: #444; color: #ffbc00; }
    QPushButton#TotalResetBtn { color: #ff453a; font-weight: bold; }
    QPushButton#TotalResetBtn:hover { background-color: #442a2a; }
"""

class SingleImageLabel(QLabel):
    file_dropped = pyqtSignal(str)
    roi_changed = pyqtSignal(QRect)

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.default_text = title
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.init_state()

    def init_state(self):
        self.orig_pixmap = None
        self.scale_factor = 1.0
        self.roi_rect = QRect(0, 0, 0, 0)
        self.laser_y, self.cutoff_y, self.std_y = -1, -1, -1
        self.active_handle = None
        self.last_mouse_pos = QPoint()
        self.setText(self.default_text)
        self.setPixmap(QPixmap())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px dashed #444; background-color: #1e1e1e; color: #888; border-radius: 10px;")

    def get_handles(self):
        if self.orig_pixmap is None: return {}
        sw, sh = self.orig_pixmap.width() / self.scale_factor, self.orig_pixmap.height() / self.scale_factor
        ox, oy = (self.width() - sw) // 2, (self.height() - sh) // 2
        r = QRect(int(self.roi_rect.left()/self.scale_factor + ox), int(self.roi_rect.top()/self.scale_factor + oy),
                  int(self.roi_rect.width()/self.scale_factor), int(self.roi_rect.height()/self.scale_factor))
        h = 14
        return {
            'top_left': QRect(r.left()-h//2, r.top()-h//2, h, h), 'top_right': QRect(r.right()-h//2, r.top()-h//2, h, h),
            'bottom_left': QRect(r.left()-h//2, r.bottom()-h//2, h, h), 'bottom_right': QRect(r.right()-h//2, r.bottom()-h//2, h, h),
            'top': QRect(r.center().x()-h//2, r.top()-h//2, h, h), 'bottom': QRect(r.center().x()-h//2, r.bottom()-h//2, h, h),
            'left': QRect(r.left()-h//2, r.center().y()-h//2, h, h), 'right': QRect(r.right()-h//2, r.center().y()-h//2, h, h)
        }

    def set_image(self, path):
        try:
            img_array = np.fromfile(path, np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None: return
            rgb_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            q_img = QImage(rgb_img.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()
            self.orig_pixmap = QPixmap.fromImage(q_img)
            self.setText(""); self.setStyleSheet("border: 1px solid #333; background-color: #000000;")
            pw, ph = self.orig_pixmap.width(), self.orig_pixmap.height()
            self.roi_rect = QRect(int(pw*0.35), int(ph*0.35), int(pw*0.3), int(ph*0.3))
            self.laser_y, self.cutoff_y, self.std_y = -1, -1, -1
            self.update_display()
            self.roi_changed.emit(self.roi_rect)
        except Exception as e: print(f"Load Error: {e}")

    def mousePressEvent(self, event):
        if self.orig_pixmap is None: return
        pos = event.position().toPoint()
        handles = self.get_handles()
        for name, rect in handles.items():
            if rect.contains(pos): self.active_handle = name; self.last_mouse_pos = pos; return
        sw, sh = self.orig_pixmap.width() / self.scale_factor, self.orig_pixmap.height() / self.scale_factor
        ox, oy = (self.width() - sw) // 2, (self.height() - sh) // 2
        r = QRect(int(self.roi_rect.left()/self.scale_factor + ox), int(self.roi_rect.top()/self.scale_factor + oy),
                  int(self.roi_rect.width()/self.scale_factor), int(self.roi_rect.height()/self.scale_factor))
        if r.contains(pos): self.active_handle = 'move'; self.last_mouse_pos = pos

    def mouseMoveEvent(self, event):
        if self.orig_pixmap is None or not self.active_handle: return
        pos = event.position().toPoint()
        diff = (pos - self.last_mouse_pos) * self.scale_factor
        new_r = QRect(self.roi_rect)
        if self.active_handle == 'move': new_r.translate(int(diff.x()), int(diff.y()))
        else:
            if 'top' in self.active_handle: new_r.setTop(new_r.top() + int(diff.y()))
            if 'bottom' in self.active_handle: new_r.setBottom(new_r.bottom() + int(diff.y()))
            if 'left' in self.active_handle: new_r.setLeft(new_r.left() + int(diff.x()))
            if 'right' in self.active_handle: new_r.setRight(new_r.right() + int(diff.x()))
        new_r = new_r.normalized()
        img_bound = QRect(0, 0, self.orig_pixmap.width(), self.orig_pixmap.height())
        if new_r.width() >= 10 and new_r.height() >= 10:
            self.roi_rect = new_r.intersected(img_bound)
            self.roi_changed.emit(self.roi_rect)
        self.last_mouse_pos = pos; self.update_display()

    def mouseReleaseEvent(self, event): self.active_handle = None

    def update_display(self, laser_y=None, cutoff_y=None, std_y=None):
        if laser_y is not None: self.laser_y = laser_y
        if cutoff_y is not None: self.cutoff_y = cutoff_y
        if std_y is not None: self.std_y = std_y
        if self.orig_pixmap is None: return

        scaled = self.orig_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.scale_factor = self.orig_pixmap.width() / scaled.width()
        ox, oy = (self.width()-scaled.width())//2, (self.height()-scaled.height())//2
        
        pix = QPixmap(self.size()); pix.fill(QColor("#121212"))
        with QPainter(pix) as p:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.drawPixmap(ox, oy, scaled)
            s_roi = QRect(int(self.roi_rect.left()/self.scale_factor)+ox, int(self.roi_rect.top()/self.scale_factor)+oy,
                          int(self.roi_rect.width()/self.scale_factor), int(self.roi_rect.height()/self.scale_factor))
            
            p.setPen(QPen(QColor(255,221,0), 2)); p.drawRect(s_roi)
            for r in self.get_handles().values(): p.setBrush(QColor(255,221,0)); p.drawRect(r)
            
            # ROI 내 수직 점선
            p.setPen(QPen(QColor(255, 59, 48, 180), 1, Qt.PenStyle.DashLine))
            p.drawLine(s_roi.center().x(), s_roi.top(), s_roi.center().x(), s_roi.bottom())
            
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            if self.laser_y != -1:
                ly = int(self.laser_y/self.scale_factor)+oy
                p.setPen(QPen(QColor(255,59,48), 2)); p.drawLine(s_roi.left(), ly, s_roi.right(), ly)
                p.drawText(s_roi.left() + 5, ly - 5, "Laser Line")
            if self.std_y != -1:
                sy = int(self.std_y/self.scale_factor)+oy
                p.setPen(QPen(QColor(10,132,255), 1, Qt.PenStyle.DashLine)); p.drawLine(s_roi.left(), sy, s_roi.right(), sy)
                p.drawText(s_roi.left() + 5, sy - 5, "1.0% Std Line")
            if self.cutoff_y != -1:
                cy = int(self.cutoff_y/self.scale_factor)+oy
                p.setPen(QPen(QColor(48,209,88), 3)); p.drawLine(s_roi.left(), cy, s_roi.right(), cy)
                p.drawText(s_roi.left() + 5, cy - 5, "Cut-off Line")
        self.setPixmap(pix)

    def dragEnterEvent(self, e): 
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e): self.file_dropped.emit(e.mimeData().urls()[0].toLocalFile())
    def resizeEvent(self, e): super().resizeEvent(e); self.update_display()

class HeadlampAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Headlamp Cut-off Analyzer")
        self.setMinimumSize(1600, 1000)
        self.setStyleSheet(DARK_STYLE)
        self.img_path = None
        
        self.device_fov_db = {
            "아이폰 15": [58.0, 31.0], "아이폰 15 플러스": [58.0, 31.0], "아이폰 15 프로": [58.0, 31.0, 12.5], "아이폰 15 프로 맥스": [58.0, 31.0, 12.5],
            "아이폰 16": [58.0, 31.0], "아이폰 16e": [58.0, 31.0], "아이폰 16 플러스": [58.0, 31.0], "아이폰 16 프로": [58.0, 31.0, 12.5], "아이폰 16 프로 맥스": [58.0, 31.0, 12.5],
            "아이폰 17": [58.0, 31.0], "아이폰 17e": [58.0, 31.0], "아이폰 17 프로": [58.0, 31.0, 12.5], "아이폰 17 프로 맥스": [58.0, 31.0, 11.0], "아이폰 Air": [58.0, 31.0],
            "갤럭시 S23": [56.2, 29.5, 17.5], "갤럭시 S23+": [56.2, 29.5, 17.5], "갤럭시 S23 울트라": [58.1, 30.5, 11.2, 6.2], "갤럭시 S23 FE": [56.2, 29.5, 17.5],
            "갤럭시 S24": [56.2, 29.5, 17.5], "갤럭시 S24+": [56.2, 29.5, 17.5], "갤럭시 S24+ 울트라": [58.1, 30.5, 17.5, 11.2], "갤럭시 S24 FE": [56.2, 29.5, 17.5],
            "갤럭시 S25": [56.5, 30.0, 17.8], "갤럭시 S25+": [56.5, 30.0, 17.8], "갤럭시 S25 엣지": [56.5, 30.0, 17.8], "갤럭시 S25 울트라": [58.1, 30.5, 11.2, 5.8], "갤럭시 S25 FE": [56.5, 30.0, 17.8],
            "갤럭시 S26": [56.5, 30.0, 17.8], "갤럭시 S26+": [56.5, 30.0, 17.8], "갤럭시 S26 울트라": [58.1, 30.5, 11.2, 5.8],
            "갤럭시 Z 플립4": [54.5, 28.0], "갤럭시 Z 플립5": [54.5, 28.0], "갤럭시 Z 플립6": [56.2, 29.5], "갤럭시 Z 플립7": [56.2, 29.5], "갤럭시 Z 플립7 FE": [56.2, 29.5],
            "갤럭시 Z 폴드4": [56.2, 29.5, 17.5], "갤럭시 Z 폴드5": [56.2, 29.5, 17.5], "갤럭시 Z 폴드6": [56.2, 29.5, 17.5], "갤럭시 Z 폴드 SE": [58.1, 30.5, 17.5], "갤럭시 Z 폴드 7": [58.1, 30.5, 17.5]
        }
        self.analysis_timer = QTimer(); self.analysis_timer.setSingleShot(True); self.analysis_timer.timeout.connect(self.run_analysis)
        self.initUI()

    def initUI(self):
        w = QWidget(); main_v = QVBoxLayout(w); main_v.setContentsMargins(15,15,15,15)
        top_h = QHBoxLayout()
        top_h.addWidget(QLabel("거리(mm):")); self.dist_in = QLineEdit("10000"); self.dist_in.setFixedWidth(70); top_h.addWidget(self.dist_in)
        top_h.addSpacing(10); top_h.addWidget(QLabel("배율:")); self.zoom_cb = QComboBox(); self.zoom_cb.setFixedWidth(120); top_h.addWidget(self.zoom_cb)
        top_h.addSpacing(15); top_h.addWidget(QLabel("밝기 Gain:")); self.gain_in = QLineEdit("1.0"); self.gain_in.setFixedWidth(50); top_h.addWidget(self.gain_in)
        top_h.addSpacing(10); top_h.addWidget(QLabel("수동 Offset(mm):")); self.offset_in = QLineEdit("0.0"); self.offset_in.setFixedWidth(60)
        self.offset_in.textChanged.connect(self.request_analysis); top_h.addWidget(self.offset_in)
        offset_btn_v = QVBoxLayout(); offset_btn_v.setSpacing(1); offset_btn_v.setContentsMargins(0,0,0,0)
        self.btn_off_up = QPushButton("▲"); self.btn_off_up.setObjectName("ArrowBtn"); self.btn_off_dn = QPushButton("▼"); self.btn_off_dn.setObjectName("ArrowBtn")
        for btn in [self.btn_off_up, self.btn_off_dn]: btn.setAutoRepeat(True); btn.setAutoRepeatDelay(300); btn.setAutoRepeatInterval(40)
        self.btn_off_up.clicked.connect(lambda: self.adjust_offset(0.1)); self.btn_off_dn.clicked.connect(lambda: self.adjust_offset(-0.1))
        offset_btn_v.addWidget(self.btn_off_up); offset_btn_v.addWidget(self.btn_off_dn); top_h.addLayout(offset_btn_v)
        
        top_h.addSpacing(20); top_h.addWidget(QLabel("기기 선택:")); self.device_cb = QComboBox(); self.device_cb.addItems(self.device_fov_db.keys())
        self.device_cb.setCurrentText("아이폰 15 프로 맥스"); self.device_cb.currentTextChanged.connect(self.update_zoom_options); top_h.addWidget(self.device_cb)
        
        top_h.addStretch()
        self.btn_upload = QPushButton("📁 이미지 불러오기"); self.btn_upload.clicked.connect(self.open_file_dialog); top_h.addWidget(self.btn_upload)
        self.btn_sub_reset = QPushButton("⚙️ Gain / Offset 초기화"); self.btn_sub_reset.setObjectName("SubResetBtn"); self.btn_sub_reset.clicked.connect(self.reset_controls); top_h.addWidget(self.btn_sub_reset)
        btn_total_r = QPushButton("전체 RESET"); btn_total_r.setObjectName("TotalResetBtn"); btn_total_r.clicked.connect(self.reset_all); top_h.addWidget(btn_total_r)
        main_v.addLayout(top_h)
        
        content_h = QHBoxLayout(); m_img_v = QVBoxLayout(); m_img_v.addWidget(QLabel("🖼️ Main View"))
        self.img_lbl = SingleImageLabel("이미지를 추가하세요"); self.img_lbl.file_dropped.connect(self.handle_file); self.img_lbl.roi_changed.connect(self.update_zoom_view)
        m_img_v.addWidget(self.img_lbl); content_h.addLayout(m_img_v, stretch=1)
        z_v = QVBoxLayout(); z_v.addWidget(QLabel("🔍 Zoom View")); self.zoom_view = QLabel("ROI Area"); self.zoom_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.zoom_view.setStyleSheet("border: 2px dashed #444; background-color: #1e1e1e; color: #888; border-radius: 10px;")
        z_v.addWidget(self.zoom_view); content_h.addLayout(z_v, stretch=1); main_v.addLayout(content_h, stretch=7)
        
        bottom_widget = QWidget(); bottom_widget.setMinimumHeight(350); bottom_h = QHBoxLayout(bottom_widget); bottom_h.setContentsMargins(0,0,0,0)
        g_v = QVBoxLayout(); g_v.addWidget(QLabel("📊 Intensity Profile")); plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 3.5), dpi=95, constrained_layout=True); self.fig.patch.set_facecolor('#121212'); self.ax.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.fig); g_v.addWidget(self.canvas); bottom_h.addLayout(g_v, stretch=3)
        res_v = QVBoxLayout(); self.btn_a = QPushButton("분석 실행 (RUN)"); self.btn_a.setFixedHeight(60); self.btn_a.setObjectName("RunBtn")
        self.btn_a.clicked.connect(self.run_analysis); res_v.addWidget(self.btn_a); self.res_lbl = QLabel("결과 대기 중")
        self.res_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); self.res_lbl.setStyleSheet("font-weight:bold; padding:10px; border:2px solid #333; background:#1e1e1e; border-radius:10px; color:#ffffff;")
        res_v.addWidget(self.res_lbl); bottom_h.addLayout(res_v, stretch=2); main_v.addWidget(bottom_widget, stretch=3); self.setCentralWidget(w)
        self.update_zoom_options()

    def update_zoom_options(self):
        self.zoom_cb.clear(); fovs = self.device_fov_db[self.device_cb.currentText()]
        labels = ["1x (Main)", "2x (Crop)", "3x (Tele)", "5x (Tele)", "10x (Tele)"]
        for i, fov in enumerate(fovs): self.zoom_cb.addItem(labels[i] if i < len(labels) else f"{i+1}x", fov)
        if len(fovs) >= 3: self.zoom_cb.setCurrentIndex(2)
        elif len(fovs) >= 1: self.zoom_cb.setCurrentIndex(0)

    def request_analysis(self): self.analysis_timer.start(30)
    def adjust_offset(self, delta):
        try:
            val_str = self.offset_in.text().strip(); curr = float(val_str) if val_str and val_str != "-" else 0.0
            self.offset_in.setText(f"{curr + delta:.1f}")
        except: self.offset_in.setText("0.0")

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path: self.handle_file(file_path)

    def handle_file(self, path): self.img_path = path; self.img_lbl.set_image(path); self.update_zoom_view(self.img_lbl.roi_rect)

    def update_zoom_view(self, r):
        if not self.img_path or self.img_lbl.orig_pixmap is None or r.isEmpty(): return
        self.zoom_view.setStyleSheet("border: 1px solid #333; background-color: #000000;")
        cropped = self.img_lbl.orig_pixmap.copy(r); scaled_zoom = cropped.scaled(self.zoom_view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        z_scale = scaled_zoom.width() / r.width(); ox, oy = (self.zoom_view.width() - scaled_zoom.width()) // 2, (self.zoom_view.height() - scaled_zoom.height()) // 2
        pix = QPixmap(self.zoom_view.size()); pix.fill(QColor("#000000"))
        with QPainter(pix) as p:
            p.setRenderHint(QPainter.RenderHint.Antialiasing); p.drawPixmap(ox, oy, scaled_zoom)
            
            # ✅ [추가] Zoom View 내 수직 점선 분석 기준선 + 텍스트 라벨
            center_x = ox + scaled_zoom.width() // 2
            p.setPen(QPen(QColor(255, 59, 48, 180), 1, Qt.PenStyle.DashLine))
            p.drawLine(center_x, oy, center_x, oy + scaled_zoom.height())
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            p.setPen(QColor(255, 59, 48, 200))
            p.drawText(center_x + 5, oy + 15, "Analysis Ref.")

            def draw_zoom_line(y_abs, color, label, width, dash=False):
                if y_abs != -1 and r.top() <= y_abs <= r.bottom():
                    y_rel = int((y_abs - r.top()) * z_scale) + oy
                    pen = QPen(QColor(color), width)
                    if dash: pen.setStyle(Qt.PenStyle.DashLine)
                    p.setPen(pen); p.drawLine(ox, y_rel, ox + scaled_zoom.width(), y_rel); p.drawText(ox + 5, y_rel - 5, label)
            draw_zoom_line(self.img_lbl.laser_y, "#ff3b30", "Laser", 2); draw_zoom_line(self.img_lbl.std_y, "#0a84ff", "1.0% Std", 1, True); draw_zoom_line(self.img_lbl.cutoff_y, "#30d158", "Cut-off", 3)
        self.zoom_view.setPixmap(pix)

    def reset_controls(self):
        self.offset_in.blockSignals(True); self.gain_in.setText("1.0"); self.offset_in.setText("0.0"); self.offset_in.blockSignals(False)
        if self.img_path: self.run_analysis()

    def reset_all(self):
        self.img_path = None; self.img_lbl.init_state(); self.zoom_view.setPixmap(QPixmap()); self.zoom_view.setText("ROI Area")
        self.zoom_view.setStyleSheet("border: 2px dashed #444; background-color: #1e1e1e; color: #888; border-radius: 10px;")
        self.ax.clear(); self.canvas.draw(); self.res_lbl.setText("결과 대기 중"); self.reset_controls()

    def run_analysis(self):
        if not self.img_path: return
        QApplication.processEvents()
        try:
            D = float(self.dist_in.text() if self.dist_in.text() else 10000.0)
            manual_offset = float(self.offset_in.text() if self.offset_in.text() and self.offset_in.text() != "-" else 0.0)
            R = self.img_lbl.roi_rect; FOV = self.zoom_cb.currentData(); gain = float(self.gain_in.text() if self.gain_in.text() else 1.0)
            img_b = cv2.imdecode(np.fromfile(self.img_path, np.uint8), cv2.IMREAD_COLOR); H_img = img_b.shape[0]
            y1, y2, x1, x2 = int(R.top()), int(R.bottom()), int(R.left()), int(R.right()); roi_b = img_b[y1:y2, x1:x2]
            roi_g = np.clip(roi_b.astype(np.float32) * gain, 0, 255).astype(np.uint8); hsv = cv2.cvtColor(roi_g, cv2.COLOR_BGR2HSV)
            m = cv2.add(cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])), cv2.add(cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255])), cv2.add(cv2.inRange(hsv, np.array([35,80,80]), np.array([90,255,255])), cv2.inRange(hsv, np.array([25,50,70]), np.array([35,255,255])))))
            if np.max(np.sum(m, axis=1)) > 50*255: l_y = np.argmax(np.sum(m, axis=1)) + y1
            else: return
            deg_per_px = FOV / H_img; std_y = l_y + (math.degrees(math.atan(0.01)) / deg_per_px)
            gray = cv2.cvtColor(roi_g, cv2.COLOR_BGR2GRAY); grad = cv2.Sobel(cv2.GaussianBlur(gray, (9,9), 0), cv2.CV_64F, 0, 1, ksize=5)
            rel_l_y = l_y - y1; grad[max(0, rel_l_y-15):min(roi_b.shape[0], rel_l_y+15), :] = 0
            c_y_auto = np.argmax(np.mean(grad, axis=1)) + y1
            c_y = c_y_auto - (math.degrees(math.atan(manual_offset / D)) / deg_per_px)
            mm_raw = D * math.tan(math.radians((l_y - c_y) * deg_per_px))
            mm_std = D * math.tan(math.radians((std_y - c_y) * deg_per_px))
            pct_raw, pct_std = (mm_raw / D) * 100, (mm_std / D) * 100
            self.img_lbl.update_display(l_y, c_y, std_y); self.update_zoom_view(self.img_lbl.roi_rect)
            
            # ✅ [유지] 그래프 라벨 및 텍스트 
            self.ax.clear(); profile = np.mean(gray, axis=1); y_idx = np.arange(y1, y2); mm_ax = [D * math.tan(math.radians((l_y - y) * deg_per_px)) for y in y_idx]
            self.ax.plot(mm_ax, profile, color='#ffffff', lw=2)
            
            y_max = np.max(profile) if len(profile) > 0 else 255
            self.ax.axvline(x=0, color='#ff3b30', lw=1.5); self.ax.text(5, y_max*0.9, "Laser", color='#ff3b30', fontweight='bold')
            self.ax.axvline(x=-(D*0.01), color='#0a84ff', linestyle='--'); self.ax.text(-(D*0.01)+5, y_max*0.8, "1.0% Std", color='#0a84ff', fontweight='bold')
            self.ax.axvline(x=mm_raw, color='#30d158', lw=2); self.ax.text(mm_raw+5, y_max*0.7, "Cut-off", color='#30d158', fontweight='bold')
            
            self.ax.xaxis.set_major_locator(ticker.MultipleLocator(50)); self.ax.grid(True, color='#555', lw=0.8); self.ax.set_xlabel("Height (mm)"); self.ax.set_ylabel("Brightness"); self.canvas.draw_idle()
            res_html = (f"<div style='line-height: 130%;'><span style='color: #888; font-size: 14px;'>[북미 사양 기준]</span><br><span style='font-size: 20px;'>{mm_raw:+.1f} mm ({pct_raw:+.2f}%)</span><br>"
                        f"<div style='margin-top: 5px;'></div><span style='color: #888; font-size: 14px;'>[유럽 사양 기준]</span><br><span style='font-size: 20px;'>{mm_std:+.1f} mm ({pct_std:+.2f}%)</span><br>"
                        f"<div style='margin-top: 5px;'></div><span style='font-size: 18px; color: {'#0a84ff' if mm_std > 0 else '#ff453a'};'>상태: {'높음(UP)' if mm_std > 0 else '낮음(DOWN)'}</span></div>")
            self.res_lbl.setText(res_html); border_color = '#0a84ff' if abs(mm_std) < 20 else '#ff453a'; self.res_lbl.setStyleSheet(f"font-weight:bold; padding:10px; background:#1e1e1e; border:3px solid {border_color}; color:#ffffff; border-radius:10px;")
        except: pass

if __name__ == "__main__":
    app = QApplication(sys.argv); window = HeadlampAnalyzer(); window.show(); sys.exit(app.exec())
