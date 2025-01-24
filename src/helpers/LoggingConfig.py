import logging as log
import logging.config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'geeksforgeeks.log',
            'formatter': 'default',
        },
        'stdout': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        'GeeksforGeeksLogger': {
            'handlers': ['file', 'stdout'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
def createLogger(name: str) -> logging.Logger:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = log.getLogger(name)
    return logger
# logger.debug('Debug message: Initializing GeeksforGeeks module.')
# logger.info('Info message: GeeksforGeeks module loaded successfully.')
# logger.warning('Warning message: GeeksforGeeks module is using deprecated functions.')
# logger.error('Error message: GeeksforGeeks module encountered an error.')
# logger.critical('Critical message: GeeksforGeeks module failed to load.')
