"""
Logging module, has everything to set ypu up to log your data to a logfile specified by LOG_FILE_PATH in constants.py.
all you need to do is:

from helpers.LoggerConfig import get_logger
log = get_logger(__name__)
...
log.info("my information I want to save")
log.debug("this is for debug purposes...")
leg.error() ...
etc...
"""

import  logging
from logging import Logger
from helpers.constants import LOG_TO_STD_OUT, LOG_FILE_PATH, LOG_LEVEL


def get_logger(module_name: str) -> Logger:
    rootLogger = logging.getLogger(module_name)
    rootLogger.setLevel(LOG_LEVEL)

    #logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logFormatter = logging.Formatter('%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s')

    fileHandler = logging.FileHandler(LOG_FILE_PATH)
    fileHandler.setLevel(LOG_LEVEL)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()#sys.stdout if LOG_TO_STD_OUT else None)
    consoleHandler.setLevel(LOG_LEVEL)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.info('created a logging config')
    print(f"Logger created with threshold: {LOG_LEVEL}, logfile will be placed in {LOG_FILE_PATH}")

    return rootLogger