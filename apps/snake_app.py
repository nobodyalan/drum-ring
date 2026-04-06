import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass, field

from PyQt6.QtCore import QTimer, Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    from common.ui_style import APP_STYLE_SHEET
except ImportError:
    from apps.common.ui_style import APP_STYLE_SHEET


STYLE_SHEET = (
    APP_STYLE_SHEET
    + """
QWidget#snakeRoot {
    background: qlineargradient(
        x1: 0, y1: 0,
        x2: 1, y2: 1,
        stop: 0 #f4f7fb,
        stop: 0.55 #eef3f8,
        stop: 1 #e8eef5
    );
}
QFrame#heroCard {
    background: rgba(255, 255, 255, 0.80);
    border: 1px solid rgba(216, 226, 237, 0.95);
    border-radius: 28px;
}
QFrame#sideCard {
    background: rgba(251, 253, 255, 0.94);
    border: 1px solid #dde6ef;
    border-radius: 22px;
}
QLabel#heroTitle {
    font-size: 28px;
    font-weight: 800;
    color: #223243;
}
QLabel#heroText {
    color: #6a7c90;
    font-size: 13px;
}
QLabel#statValue {
    color: #213446;
    font-size: 28px;
    font-weight: 800;
}
QLabel#statLabel {
    color: #7d8fa3;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}
QLabel#hintPill {
    background: rgba(243, 247, 251, 0.95);
    border: 1px solid #dee7f0;
    border-radius: 14px;
    padding: 8px 12px;
    color: #6a7c90;
}
QPushButton#ghostButton {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #d8e2ec;
    border-radius: 14px;
    padding: 10px 16px;
    color: #4b6177;
    font-weight: 700;
}
QPushButton#ghostButton:hover {
    background: #ffffff;
    border: 1px solid #c8d5e2;
}
QPushButton#ghostButton:pressed {
    background: #eff4f8;
}
QSlider#speedSlider::groove:horizontal {
    height: 8px;
    background: #e2eaf3;
    border-radius: 4px;
}
QSlider#speedSlider::sub-page:horizontal {
    background: #a8cce0;
    border-radius: 4px;
}
QSlider#speedSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #9bbfd5;
    width: 18px;
    margin: -7px 0;
    border-radius: 9px;
}
"""
)

BOARD_COLUMNS = 18
BOARD_ROWS = 18
INITIAL_SPEED = 8
MIN_SPEED = 4
MAX_SPEED = 18


@dataclass
class SnakePalette:
    board: QColor = field(default_factory=lambda: QColor("#fbfdff"))
    grid_a: QColor = field(default_factory=lambda: QColor("#f1f5f9"))
    grid_b: QColor = field(default_factory=lambda: QColor("#edf2f7"))
    border: QColor = field(default_factory=lambda: QColor("#dbe5ef"))
    snake_head: QColor = field(default_factory=lambda: QColor("#234456"))
    snake_body_start: QColor = field(default_factory=lambda: QColor("#4e8ea7"))
    snake_body_end: QColor = field(default_factory=lambda: QColor("#9cc9d8"))
    food_outer: QColor = field(default_factory=lambda: QColor("#ffb8a6"))
    food_inner: QColor = field(default_factory=lambda: QColor("#ff8668"))
    overlay: QColor = field(default_factory=lambda: QColor(255, 255, 255, 168))
    overlay_text: QColor = field(default_factory=lambda: QColor("#223243"))
    overlay_sub_text: QColor = field(default_factory=lambda: QColor("#6a7c90"))


class SnakeBoard(QWidget):
    def __init__(self, window: "SnakeGameWindow"):
        super().__init__(window)
        self.game_window = window
        self.palette = SnakePalette()
        self.setMinimumSize(560, 560)

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        outer_rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(self.palette.border, 1.2))
        painter.setBrush(self.palette.board)
        painter.drawRoundedRect(outer_rect, 26, 26)

        board_size = min(outer_rect.width(), outer_rect.height()) - 28
        board_left = outer_rect.center().x() - board_size / 2
        board_top = outer_rect.center().y() - board_size / 2
        board_rect = QRectF(board_left, board_top, board_size, board_size)

        cell_size = board_rect.width() / BOARD_COLUMNS
        inset = max(2.4, cell_size * 0.11)
        radius = max(4.0, cell_size * 0.22)

        self._paint_grid(painter, board_rect, cell_size, inset, radius)
        self._paint_food(painter, board_rect, cell_size, inset, radius)
        self._paint_snake(painter, board_rect, cell_size, inset, radius)
        self._paint_overlay(painter, board_rect)

    def _paint_grid(
        self, painter: QPainter, board_rect: QRectF, cell_size: float, inset: float, radius: float
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLUMNS):
                color = self.palette.grid_a if (row + col) % 2 == 0 else self.palette.grid_b
                painter.setBrush(color)
                cell = QRectF(
                    board_rect.left() + col * cell_size + inset,
                    board_rect.top() + row * cell_size + inset,
                    cell_size - inset * 2,
                    cell_size - inset * 2,
                )
                painter.drawRoundedRect(cell, radius, radius)

    def _paint_food(
        self, painter: QPainter, board_rect: QRectF, cell_size: float, inset: float, radius: float
    ) -> None:
        food = self.game_window.food
        if food is None:
            return

        col, row = food
        outer = QRectF(
            board_rect.left() + col * cell_size + inset,
            board_rect.top() + row * cell_size + inset,
            cell_size - inset * 2,
            cell_size - inset * 2,
        )
        inner_margin = outer.width() * 0.2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette.food_outer)
        painter.drawRoundedRect(outer, radius, radius)
        painter.setBrush(self.palette.food_inner)
        painter.drawRoundedRect(
            outer.adjusted(inner_margin, inner_margin, -inner_margin, -inner_margin),
            max(3.0, radius * 0.65),
            max(3.0, radius * 0.65),
        )

    def _paint_snake(
        self, painter: QPainter, board_rect: QRectF, cell_size: float, inset: float, radius: float
    ) -> None:
        snake = self.game_window.snake
        if not snake:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        last_index = max(1, len(snake) - 1)
        for index, (col, row) in enumerate(reversed(snake)):
            blend = index / last_index
            color = self._blend_color(
                self.palette.snake_head if index == 0 else self.palette.snake_body_start,
                self.palette.snake_body_end,
                0.0 if index == 0 else min(0.82, blend),
            )
            painter.setBrush(color)
            cell = QRectF(
                board_rect.left() + col * cell_size + inset,
                board_rect.top() + row * cell_size + inset,
                cell_size - inset * 2,
                cell_size - inset * 2,
            )
            shrink = 0.0 if index == 0 else cell.width() * 0.05
            painter.drawRoundedRect(
                cell.adjusted(shrink, shrink, -shrink, -shrink),
                radius,
                radius,
            )

    def _paint_overlay(self, painter: QPainter, board_rect: QRectF) -> None:
        title = ""
        subtitle = ""
        if self.game_window.game_over:
            title = "游戏结束"
            subtitle = "按空格开始新一局"
        elif not self.game_window.started:
            title = "准备开始"
            subtitle = "先用方向键预输入，再按空格开始"
        elif self.game_window.paused:
            title = "已暂停"
            subtitle = "按空格继续"

        if not title:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette.overlay)
        painter.drawRoundedRect(board_rect, 24, 24)

        title_font = QFont()
        title_font.setPointSize(19)
        title_font.setWeight(700)
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)

        painter.setPen(self.palette.overlay_text)
        painter.setFont(title_font)
        painter.drawText(board_rect.adjusted(0, -18, 0, 0), Qt.AlignmentFlag.AlignCenter, title)

        painter.setPen(self.palette.overlay_sub_text)
        painter.setFont(subtitle_font)
        painter.drawText(board_rect.adjusted(0, 28, 0, 0), Qt.AlignmentFlag.AlignCenter, subtitle)

    @staticmethod
    def _blend_color(start: QColor, end: QColor, ratio: float) -> QColor:
        ratio = max(0.0, min(1.0, ratio))
        return QColor(
            int(start.red() + (end.red() - start.red()) * ratio),
            int(start.green() + (end.green() - start.green()) * ratio),
            int(start.blue() + (end.blue() - start.blue()) * ratio),
        )


class SnakeGameWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake")
        self.resize(1080, 760)
        self.setMinimumSize(900, 680)
        self.setObjectName("snakeRoot")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)

        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] | None = None
        self.direction = (1, 0)
        self.input_buffer: deque[tuple[int, int]] = deque(maxlen=3)
        self.staged_input_buffer: deque[tuple[int, int]] = deque(maxlen=3)
        self.score = 0
        self.best_score = 0
        self.started = False
        self.paused = False
        self.game_over = False

        self.board = SnakeBoard(self)
        self.score_value = QLabel()
        self.best_value = QLabel()
        self.speed_value = QLabel()
        self.status_tag = QLabel()
        self.pause_button = QPushButton("暂停")
        self.restart_button = QPushButton("重新开始")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.status_tag.setObjectName("statusTag")

        self._build_ui()
        self._bind_signals()
        self._reset_game()
        self._apply_speed()
        QTimer.singleShot(0, self.setFocus)

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(26, 26, 26, 26)
        root.setSpacing(18)

        left_card = QFrame()
        left_card.setObjectName("heroCard")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(22, 22, 22, 22)
        left_layout.setSpacing(16)

        title = QLabel("Snake")
        title.setObjectName("heroTitle")
        desc = QLabel("简洁方格棋盘，支持预输入转向，空格负责开始、暂停和继续。")
        desc.setObjectName("heroText")

        header_row = QHBoxLayout()
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self.status_tag)

        hint = QLabel("控制提示: ↑ ↓ ← → 预输入方向 | Space 开始 / 暂停 / 继续")
        hint.setObjectName("hintPill")

        left_layout.addLayout(header_row)
        left_layout.addWidget(desc)
        left_layout.addWidget(hint)
        left_layout.addWidget(self.board, stretch=1)

        right_card = QFrame()
        right_card.setObjectName("sideCard")
        right_card.setFixedWidth(290)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(18)

        right_layout.addWidget(self._create_stat_card("当前分数", self.score_value))
        right_layout.addWidget(self._create_stat_card("最高分", self.best_value))
        right_layout.addWidget(self._create_stat_card("移动速度", self.speed_value))

        speed_title = QLabel("速度调节")
        speed_title.setProperty("class", "settings-title")
        speed_desc = QLabel("拖动滑杆调整每秒移动格数。")
        speed_desc.setProperty("class", "settings-desc")
        self.speed_slider.setObjectName("speedSlider")
        self.speed_slider.setRange(MIN_SPEED, MAX_SPEED)
        self.speed_slider.setValue(INITIAL_SPEED)
        self.speed_slider.setSingleStep(1)
        self.speed_slider.setPageStep(2)

        pause_hint = QLabel("支持两级方向缓存。高速连续转向时，会优先保留后续输入。")
        pause_hint.setObjectName("hintPill")
        pause_hint.setWordWrap(True)

        self.pause_button.setObjectName("ghostButton")
        self.restart_button.setObjectName("ghostButton")

        right_layout.addWidget(speed_title)
        right_layout.addWidget(speed_desc)
        right_layout.addWidget(self.speed_slider)
        right_layout.addWidget(pause_hint)
        right_layout.addStretch()
        right_layout.addWidget(self.pause_button)
        right_layout.addWidget(self.restart_button)

        root.addWidget(left_card, stretch=1)
        root.addWidget(right_card)

    def _create_stat_card(self, label_text: str, value_widget: QLabel) -> QFrame:
        card = QFrame()
        card.setObjectName("sideCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        label = QLabel(label_text)
        label.setObjectName("statLabel")
        value_widget.setObjectName("statValue")

        layout.addWidget(label)
        layout.addWidget(value_widget)
        return card

    def _bind_signals(self) -> None:
        self.pause_button.clicked.connect(self.toggle_pause)
        self.restart_button.clicked.connect(self._reset_game)
        self.speed_slider.valueChanged.connect(self._apply_speed)

    def _reset_game(self) -> None:
        center_x = BOARD_COLUMNS // 2
        center_y = BOARD_ROWS // 2
        self.snake = [
            (center_x - 2, center_y),
            (center_x - 1, center_y),
            (center_x, center_y),
        ]
        self.direction = (1, 0)
        self.input_buffer.clear()
        self.staged_input_buffer.clear()
        self.score = 0
        self.started = False
        self.paused = False
        self.game_over = False
        self.timer.stop()
        self._spawn_food()
        self._refresh_ui()
        self.board.update()
        self.setFocus()

    def _spawn_food(self) -> None:
        empty_cells = [
            (col, row)
            for row in range(BOARD_ROWS)
            for col in range(BOARD_COLUMNS)
            if (col, row) not in self.snake
        ]
        self.food = random.choice(empty_cells) if empty_cells else None

    def _apply_speed(self) -> None:
        speed = self.speed_slider.value()
        self.timer.setInterval(max(45, int(1000 / speed)))
        self.speed_value.setText(f"{speed} 格 / 秒")

    def toggle_pause(self) -> None:
        if self.game_over:
            self._reset_game()
            self.started = True
            self.timer.start()
            self._refresh_ui()
            self.board.update()
            self.setFocus()
            return
        if not self.started:
            self.started = True
            self.timer.start()
            self._refresh_ui()
            self.board.update()
            self.setFocus()
            return
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
        else:
            self.timer.start()
        self._refresh_ui()
        self.board.update()
        self.setFocus()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_pause()
            return

        direction_map = {
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
        }
        next_direction = direction_map.get(key)
        if next_direction is None:
            super().keyPressEvent(event)
            return

        if self.game_over:
            return

        queued_direction = self._last_buffered_direction()
        current_dx, current_dy = queued_direction if queued_direction is not None else self.direction
        next_dx, next_dy = next_direction
        if (current_dx + next_dx, current_dy + next_dy) == (0, 0):
            return

        if queued_direction == next_direction:
            return

        if len(self.input_buffer) < self.input_buffer.maxlen:
            self.input_buffer.append(next_direction)
        elif len(self.staged_input_buffer) < self.staged_input_buffer.maxlen:
            self.staged_input_buffer.append(next_direction)
        else:
            self.staged_input_buffer.popleft()
            self.staged_input_buffer.append(next_direction)
        self._refresh_ui()
        self.board.update()

    def _advance(self) -> None:
        if not self.started or self.paused or self.game_over:
            return

        self._promote_staged_inputs()
        if self.input_buffer:
            self.direction = self.input_buffer.popleft()
            self._promote_staged_inputs()
        next_head = (
            (self.snake[-1][0] + self.direction[0]) % BOARD_COLUMNS,
            (self.snake[-1][1] + self.direction[1]) % BOARD_ROWS,
        )

        body_to_check = self.snake if next_head == self.food else self.snake[1:]
        if next_head in body_to_check:
            self.game_over = True
            self.best_score = max(self.best_score, self.score)
            self.timer.stop()
            self._refresh_ui()
            self.board.update()
            return

        self.snake.append(next_head)
        if next_head == self.food:
            self.score += 1
            self.best_score = max(self.best_score, self.score)
            self._spawn_food()
        else:
            self.snake.pop(0)

        if self.food is None:
            self.game_over = True
            self.best_score = max(self.best_score, self.score)
            self.timer.stop()

        self._refresh_ui()
        self.board.update()

    def _promote_staged_inputs(self) -> None:
        while len(self.input_buffer) < self.input_buffer.maxlen and self.staged_input_buffer:
            direction = self.staged_input_buffer.popleft()
            last_direction = self._last_buffered_direction()
            if last_direction == direction:
                continue
            self.input_buffer.append(direction)

    def _last_buffered_direction(self) -> tuple[int, int] | None:
        if self.staged_input_buffer:
            return self.staged_input_buffer[-1]
        if self.input_buffer:
            return self.input_buffer[-1]
        return None

    def _refresh_ui(self) -> None:
        self.score_value.setText(str(self.score))
        self.best_value.setText(str(self.best_score))

        if self.game_over:
            status = "本局结束"
            pause_text = "重新开始"
        elif not self.started:
            status = "等待开始"
            pause_text = "开始"
        elif self.paused:
            status = "已暂停"
            pause_text = "继续"
        else:
            status = "进行中"
            pause_text = "暂停"

        self.status_tag.setText(status)
        self.pause_button.setText(pause_text)
        self.pause_button.setEnabled(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17888)
    return parser.parse_args()


def main() -> int:
    _ = parse_args()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = SnakeGameWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
