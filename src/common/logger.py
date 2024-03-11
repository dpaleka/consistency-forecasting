import logging


def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("forecasting.log"),
                            logging.StreamHandler()
                        ])

def get_logger(name):
    setup_logging()
    return logging.getLogger(name)
