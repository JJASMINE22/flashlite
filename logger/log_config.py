# -*- coding: UTF-8 -*-
import logging
import os
import os.path
import sys
from .log_file_handler import LogFileHandler


# 获取当前绝对路径
def get_cwd():
    logs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/logs"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    return logs_path


def log_config(log_name):
    log_path = os.path.join(get_cwd(), log_name)
    fmt = "%(asctime)s %(levelname)s [%(filename)s]: %(lineno)s - %(funcName)s - %(message)s"
    consoleHandler = logging.StreamHandler(sys.stdout)
    fileHandler = LogFileHandler(log_path)
    logging.basicConfig(level="INFO", format=fmt, datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[consoleHandler, fileHandler])
    logging.getLogger(__name__).setLevel("INFO")
