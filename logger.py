import logging
import os
from datetime import datetime


def create_logger(module_name):
    log_path = os.path.join('./',"logs", module_name)
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    log_file_path = os.path.join(log_path, f"{module_name}.log")
    formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__=='__main__':
    log = create_logger('demo')
    log.info('Logs Created')
