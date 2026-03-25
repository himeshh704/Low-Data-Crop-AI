import logging
import os
import sys

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s: %(message)s')

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, "crop_ai.log"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            # Fallback if there are permission issues creating logs dir
            print(f"Warning: Failed to setup FileHandler for logger. {e}")

    return logger
