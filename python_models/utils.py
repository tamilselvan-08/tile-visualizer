import os
import hashlib
import logging
import pickle
from PIL import Image

def setup_logger(name="room_visualizer", level=logging.INFO):
    """Sets up a standardized logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

class CacheManager:
    """Intelligent caching for AI inference and intermediate results."""
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_hash(self, *args):
        """Generates a stable hash for a set of inputs."""
        hasher = hashlib.md5()
        for arg in args:
            if isinstance(arg, Image.Image):
                hasher.update(arg.tobytes())
            elif isinstance(arg, str):
                hasher.update(arg.encode('utf-8'))
            elif hasattr(arg, 'tobytes'):
                hasher.update(arg.tobytes())
            else:
                hasher.update(str(arg).encode('utf-8'))
        return hasher.hexdigest()

    def get(self, key_prefix, *hash_args):
        key = f"{key_prefix}_{self._get_hash(*hash_args)}.pkl"
        path = os.path.join(self.cache_dir, key)
        if os.path.exists(path):
            logger.info(f"Cache hit for {key_prefix}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, key_prefix, data, *hash_args):
        key = f"{key_prefix}_{self._get_hash(*hash_args)}.pkl"
        path = os.path.join(self.cache_dir, key)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Cached {key_prefix} to {path}")

cache = CacheManager()
