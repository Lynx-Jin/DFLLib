"""Create one self-contained directory and terminal transcript per experiment."""

import atexit
import json
import os
import re
import sys
import threading
from datetime import datetime


def _safe(value):
    text = str(value).strip().replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)


def build_experiment_name(args, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    actual_rounds = int(args.global_rounds) + 1
    malicious_clients = int(args.num_clients * args.attack_malicious_ratio)
    fields = [
        timestamp,
        _safe(args.dataset),
        _safe(args.algorithm),
        _safe(args.model),
        f"nc{args.num_clients}",
        f"mc{malicious_clients}",
        f"jr{args.join_ratio:g}",
        f"r{actual_rounds}",
        f"le{args.local_epochs}",
        f"atk-{_safe(args.attack_type)}",
        f"dev-{_safe(args.attack_deviation)}",
        f"seed{args.attack_seed}",
    ]
    return "_".join(fields)


class _TeeStream:
    def __init__(self, terminal, transcript, lock):
        self.terminal = terminal
        self.transcript = transcript
        self.lock = lock

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.transcript.write(text)
            self.transcript.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.transcript.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class ExperimentLogger:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.transcript_path = os.path.join(
            experiment_dir, "terminal_output.txt")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._file = open(
            self.transcript_path, "w", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self.stdout = _TeeStream(self._stdout, self._file, self._lock)
        self.stderr = _TeeStream(self._stderr, self._file, self._lock)
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self._closed = False
        atexit.register(self.close)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if sys.stdout is self.stdout:
            sys.stdout = self._stdout
        if sys.stderr is self.stderr:
            sys.stderr = self._stderr
        self._file.flush()
        self._file.close()


def setup_experiment_logger(args):
    experiment_name = build_experiment_name(args)
    experiment_dir = os.path.abspath(
        os.path.join(args.logger_root, experiment_name))
    os.makedirs(experiment_dir, exist_ok=False)

    args.experiment_name = experiment_name
    args.experiment_log_dir = experiment_dir
    args.save_folder_name = os.path.join(experiment_dir, "items")
    monitor_subdir = os.path.basename(
        os.path.normpath(getattr(args, "attack_monitor_dir", "attack_monitor")))
    args.attack_monitor_dir = os.path.join(experiment_dir, monitor_subdir)

    os.makedirs(args.save_folder_name, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.json"),
              "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False, default=str)

    logger = ExperimentLogger(experiment_dir)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Terminal transcript: {logger.transcript_path}")
    return logger
