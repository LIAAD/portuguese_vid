import logging
import os


def setup_logger(CURRENT_PATH, CURRENT_TIME):
    print(f"Logging to {os.path.join(CURRENT_PATH, 'out', str(CURRENT_TIME), 'logs', 'log.txt')}")
    logging.basicConfig(
        filename=os.path.join(CURRENT_PATH, "out", str(CURRENT_TIME), "logs", "log.txt"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def create_output_dir(CURRENT_PATH, CURRENT_TIME):
    os.mkdir(os.path.join(CURRENT_PATH, "out", str(CURRENT_TIME)))
    os.mkdir(os.path.join(CURRENT_PATH, "out", str(CURRENT_TIME), "logs"))
    os.mkdir(os.path.join(CURRENT_PATH, "out", str(CURRENT_TIME), "models"))
