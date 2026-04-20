import sys
import os

os.environ["QT_API"] = "pyqt6"
import asyncio

from qasync import QEventLoop

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from apps.framework.ui_style import MAIN_STYLE_SHEET
from desktop.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(MAIN_STYLE_SHEET)

    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    win = MainWindow()
    win.show()
    asyncio.get_event_loop().create_task(win.start_background_services())

    with loop:
        loop.run_forever()
