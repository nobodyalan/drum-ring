BASE_STYLE_SHEET = """
QMainWindow {
    background-color: #f5f7fa;
}
QWidget {
    color: #4f6074;
    background-color: #f5f7fa;
}

QMenuBar,
QMenu,
QAbstractScrollArea,
QTextEdit,
QPlainTextEdit,
QListView,
QTreeView,
QTableView,
QScrollArea,
QAbstractItemView {
    background-color: #ffffff;
    color: #4f6074;
    alternate-background-color: #f8fbfe;
    selection-background-color: #dbe9f6;
    selection-color: #2f3e52;
}

/* ---------- Panels ---------- */
QFrame.panel {
    background-color: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 18px;
}
QFrame.control-card {
    background-color: #fbfcfe;
    border: 1px solid #e7edf4;
    border-radius: 14px;
}
QFrame.settings-panel {
    background-color: #fbfcfe;
    border: 1px solid #e7edf4;
    border-radius: 16px;
}

/* ---------- Text ---------- */
QLabel {
    font-size: 13px;
    color: #4f6074;
}
QLabel.title {
    font-size: 20px;
    font-weight: 700;
    color: #2f3e52;
}
QLabel.sub-text {
    font-size: 12px;
    color: #7f90a4;
}
QLabel#statusTag {
    background: #f3f6fa;
    border: 1px solid #dce5ee;
    border-radius: 10px;
    padding: 7px 12px;
    color: #687b90;
}
QLabel#valuePill {
    background: #f2f5f9;
    border-radius: 11px;
    padding: 4px 10px;
    font-weight: 600;
    color: #5f7186;
}

/* ---------- Buttons ---------- */
QPushButton {
    border-radius: 12px;
    padding: 8px 16px;
    border: 1px solid #d7e0ea;
    font-weight: 600;
    color: #506174;
    min-height: 20px;
    font-size: 13px;
    background: #f8fafd;
}
QPushButton:hover {
    background: #f1f5fa;
    border: 1px solid #ccd8e5;
}
QPushButton:pressed {
    background: #e8eef5;
    border: 1px solid #bfccda;
}
QPushButton:disabled {
    color: #a8b5c2;
    background: #f7f9fc;
    border: 1px solid #e5ebf2;
}

QPushButton.btn-primary {
    color: #44627f;
    background: #e8f1fa;
    border: 1px solid #cfe0f0;
}
QPushButton.btn-primary:hover {
    background: #deebf7;
    border: 1px solid #bfd6ea;
}
QPushButton.btn-primary:pressed {
    background: #d2e3f2;
    border: 1px solid #b0cae1;
}
QPushButton.btn-primary:disabled {
    color: #a7b6c5;
    background: #f0f5fa;
    border: 1px solid #e0e9f2;
}

QPushButton.btn-success {
    color: #587569;
    background: #edf6f1;
    border: 1px solid #d6e8de;
}
QPushButton.btn-success:hover {
    background: #e4f1ea;
    border: 1px solid #c7ddd1;
}
QPushButton.btn-success:pressed {
    background: #d8e9e0;
    border: 1px solid #b7d1c3;
}
QPushButton.btn-success:disabled {
    color: #a8b7b0;
    background: #f3f8f5;
    border: 1px solid #e3ede8;
}

QPushButton.btn-danger {
    color: #8d6666;
    background: #faefef;
    border: 1px solid #edd9d9;
}
QPushButton.btn-danger:hover {
    background: #f7e5e5;
    border: 1px solid #e5cbcb;
}
QPushButton.btn-danger:pressed {
    background: #f0d9d9;
    border: 1px solid #d9bbbb;
}
QPushButton.btn-danger:disabled {
    color: #baa7a7;
    background: #fcf5f5;
    border: 1px solid #f0e4e4;
}

QPushButton.btn-gray {
    color: #526375;
    background: #f2f5f9;
    border: 1px solid #d7e0ea;
}
QPushButton.btn-gray:hover {
    background: #ebf0f6;
    border: 1px solid #ccd7e3;
}
QPushButton.btn-gray:pressed {
    background: #e2e9f1;
    border: 1px solid #bfccda;
}
QPushButton.btn-gray:disabled {
    color: #a8b5c2;
    background: #f7f9fc;
    border: 1px solid #e5ebf2;
}

QPushButton.segment {
    color: #66798d;
    background: #ffffff;
    border: 1px solid #dbe4ed;
    min-height: 36px;
    padding: 0 14px;
}
QPushButton.segment:hover {
    background: #f8fafd;
    border: 1px solid #cfdae6;
}
QPushButton.segment:checked {
    color: #44627f;
    background: #e8f1fa;
    border: 1px solid #cfe0f0;
}
QPushButton.segment:pressed {
    background: #d7e6f4;
    border: 1px solid #b8cde1;
}

/* ---------- CheckBox ---------- */
QCheckBox {
    font-size: 13px;
    color: #4f6074;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #cfdbe7;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #d5e6f5;
    border: 1px solid #bcd5ea;
}
QCheckBox::indicator:disabled {
    background: #eef3f8;
    border: 1px solid #e0e7ef;
}

/* ---------- Inputs ---------- */
QSpinBox,
QDoubleSpinBox,
QComboBox {
    background-color: #ffffff;
    border: 1px solid #d9e2eb;
    border-radius: 10px;
    padding: 7px 10px;
    min-height: 20px;
    color: #45576a;
    font-size: 13px;
}
QDoubleSpinBox#settingsField,
QComboBox#settingsField {
    min-height: 36px;
    border-radius: 12px;
    padding: 0 12px;
    background-color: #ffffff;
}
QComboBox#settingsField::drop-down {
    border: none;
    width: 28px;
}
QComboBox#settingsField::down-arrow {
    width: 10px;
    height: 10px;
}
QLabel.settings-title {
    font-size: 13px;
    font-weight: 600;
    color: #304055;
}
QLabel.settings-desc {
    font-size: 12px;
    color: #8090a4;
}
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {
    border: 1px solid #b9d3ea;
    background: #fcfdff;
}
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {
    background-color: #f4f7fa;
    color: #aab7c4;
}

/* ---------- Scroll ---------- */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 4px 0 4px 0;
}
QScrollBar::handle:vertical {
    background: #d2ddea;
    border-radius: 6px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover {
    background: #c3d0de;
}
QScrollBar:horizontal {
    background: transparent;
    height: 14px;
    margin: 6px 20px 0 20px;
}
QScrollBar::handle:horizontal {
    background: #d2ddea;
    border-radius: 7px;
    min-width: 48px;
}
QScrollBar::handle:horizontal:hover {
    background: #c3d0de;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    border: none;
    background: none;
    width: 0px;
    height: 0px;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: #eef3f8;
    border-radius: 7px;
}

/* ---------- Slider ---------- */
QSlider::groove:horizontal {
    border: 0px;
    height: 10px;
    background: #e6edf5;
    border-radius: 5px;
}
QSlider::sub-page:horizontal {
    background: #d3e2f1;
    border-radius: 5px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #bfd5e8;
    width: 20px;
    margin: -7px 0;
    border-radius: 10px;
}

/* ---------- Progress ---------- */
QProgressBar {
    border: none;
    border-radius: 8px;
    background: #e8eef5;
    text-align: center;
    color: #506174;
    min-height: 16px;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: #cddff0;
}
"""

MAIN_EXTRA_STYLE_SHEET = """
QFrame[class="device-card"] {
    background-color: #ffffff;
    border: 1px solid #e2e9f0;
    border-radius: 16px;
}
QFrame[class="device-card-connected"] {
    background-color: #f1f7fd;
    border: 1px solid #d6e5f2;
    border-radius: 16px;
}
QLabel.stat-tag {
    background: #f2f5f9;
    border-radius: 10px;
    padding: 4px 10px;
    color: #5f7186;
    font-weight: 600;
}
"""

APP_STYLE_SHEET = BASE_STYLE_SHEET
MAIN_STYLE_SHEET = BASE_STYLE_SHEET + "\n" + MAIN_EXTRA_STYLE_SHEET
