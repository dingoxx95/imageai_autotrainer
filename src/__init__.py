import os
from .ml_logger import MLLogger

__version__ = '0.1.0'


# Crea le directory necessarie
REQUIRED_DIRS = ['logs', 'models', 'data', 'training_checkpoints']
for dir_name in REQUIRED_DIRS:
    os.makedirs(dir_name, exist_ok=True)

MODEL_CONFIG = {
    'INPUT_SHAPE': (32, 32, 3),
    'NUM_CLASSES': 10,
    'BATCH_SIZE': 32,
    'EPOCHS': 10
}

# Setup logging base
default_logger = MLLogger()

# Funzione per cambiare il logger di default
def set_logger(new_logger):
    global default_logger
    default_logger = new_logger

from .gpu_config import GPUConfig
from .cifar_dataset import CifarDataset
from .cnn_model import CNNModel
from .model_trainer import ModelTrainer
from .visualizer import Visualizer
from .ml_logger import MLLogger
from .model_manager import ModelManager
from .ui_manager import TrainingUI

__all__ = [
    'GPUConfig',
    'CifarDataset',
    'CNNModel',
    'ModelTrainer',
    'Visualizer',
    'MLLogger',
    'ModelManager',
    'default_logger',
    'MODEL_CONFIG',
    'TrainingUI'
]