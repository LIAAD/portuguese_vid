import logging
import os


def setup_logger(current_path, current_time):
    print(f"Logging to {os.path.join(current_path, 'out', str(current_time), 'logs', 'log.txt')}")
    logging.basicConfig(
        filename=os.path.join(current_path, "out", str(current_time), "logs", "log.txt"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def create_output_dir(current_path, current_time):
    os.mkdir(os.path.join(current_path, "out", str(current_time)))
    os.mkdir(os.path.join(current_path, "out", str(current_time), "logs"))
    os.mkdir(os.path.join(current_path, "out", str(current_time), "models"))
