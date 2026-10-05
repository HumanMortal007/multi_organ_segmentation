"""
utils/timing.py
================
Training time tracking and estimation utilities.
"""

import time
import math


class Timer:
    """Tracks epoch times and estimates remaining training time."""

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.experiment_start = None
        self.epoch_start = None

    def start_experiment(self):
        self.experiment_start = time.time()

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self):
        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        return elapsed

    def avg_epoch_time(self):
        if not self.epoch_times:
            return 0.0
        # Use last 3 epochs for a more stable estimate
        return sum(self.epoch_times[-3:]) / len(self.epoch_times[-3:])

    def estimated_remaining(self, completed_epochs):
        remaining = self.total_epochs - completed_epochs
        return self.avg_epoch_time() * remaining

    def total_elapsed(self):
        return time.time() - self.experiment_start

    @staticmethod
    def format_time(seconds):
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"

    def print_estimate(self, completed_epochs):
        """Print time estimates after a few epochs have run."""
        avg = self.avg_epoch_time()
        remaining_s = self.estimated_remaining(completed_epochs)
        full_run_20 = avg * 20
        full_run_50 = avg * 50

        print(f"  Average Epoch Time   : {self.format_time(avg)}")
        print(f"  Estimated Remaining  : {self.format_time(remaining_s)}")
        print(f"  Est. Time for 20 ep  : {self.format_time(full_run_20)}")
        print(f"  Est. Time for 50 ep  : {self.format_time(full_run_50)}")
