import logging
import sys
from datetime import datetime


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        if not datefmt:
            return super().formatTime(record, datefmt=datefmt)

        return datetime.fromtimestamp(record.created).astimezone().strftime(datefmt)

formatter = MicrosecondFormatter('%(asctime)s - %(levelname)s - %(message)s', 
    datefmt="%Y-%m-%d %H:%M:%S.%f")

file_handler = logging.FileHandler('debug.log')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)