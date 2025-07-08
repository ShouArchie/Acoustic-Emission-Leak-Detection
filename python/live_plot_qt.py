import os
import sys
from typing import List

import numpy as np
import psycopg2

from PyQt6 import QtWidgets
import pyqtgraph as pg

DSN = os.getenv("DATABASE_URL", "postgres://pico:pass@localhost:5432/piezo_data")

# --- configuration ---
WINDOW_SEC = 5        # how many seconds to display
BATCH_PERIOD = 0.01   # seconds per batch (2 000 samples @ 200 kHz)
NBATCHES = int(WINDOW_SEC / BATCH_PERIOD)  # 500


def get_connection():
    return psycopg2.connect(DSN)


def fetch_window(cur) -> np.ndarray:
    cur.execute("SELECT voltages FROM batches ORDER BY id DESC LIMIT %s;", (NBATCHES,))
    rows = cur.fetchall()
    if not rows:
        return np.array([], dtype=np.float32)
    # rows are newest→oldest; reverse and flatten
    data = np.hstack([row[0] for row in reversed(rows)]).astype(np.float32)
    return data


def main():
    conn = get_connection()
    cur = conn.cursor()

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title="Live Piezo Batch @200kHz")
    plot = win.addPlot(row=0, col=0)
    plot.setLabel("left", "Voltage", units="V")
    plot.setLabel("bottom", "Sample")
    curve = plot.plot(pen=pg.mkPen(color="c", width=1))
    plot.setYRange(0, 3.3)

    timer = pg.QtCore.QTimer()
    timer.setInterval(100)  # 10 fps

    def update():
        data = fetch_window(cur)
        if data.size == 0:
            return
        x = np.arange(data.size)
        curve.setData(x, data, downsample=8, autoDownsample=True)

    timer.timeout.connect(update)
    timer.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main() 