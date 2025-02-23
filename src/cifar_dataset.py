from . import default_logger, MODEL_CONFIG
from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
import pickle
import os
from typing import Tuple
from .ml_logger import MLLogger

@dataclass(slots=True)
class CifarDataset:
    train_images: np.ndarray = field(default_factory=lambda: np.array([]))
    train_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    test_images: np.ndarray = field(default_factory=lambda: np.array([]))
    test_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    path: str = 'data/cifar10_data.pkl'
    num_classes: int = MODEL_CONFIG['NUM_CLASSES']
    logger: MLLogger = field(default_factory=lambda: default_logger)
    
    def load(self) -> None:
        if os.path.exists(self.path):
            self._load_from_pickle()
        else:
            self._download_and_save()
        
        # Log dataset info
        dataset_info = {
            "train_images_shape": self.train_images.shape,
            "test_images_shape": self.test_images.shape,
            "num_classes": self.num_classes,
            "data_type": str(self.train_images.dtype),
            "storage_path": self.path
        }
        self.logger.log_dataset_info(dataset_info)
    
    def _normalize_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        images = images.astype('float32') / 255
        labels = tf.keras.utils.to_categorical(labels, self.num_classes)
        self.logger.log_info("Data normalized: images converted to float32 and scaled to [0,1], labels one-hot encoded")
        return images, labels
    
    def _load_from_pickle(self) -> None:
        try:
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
                self.train_images = data['train_images']
                self.train_labels = data['train_labels']
                self.test_images = data['test_images']
                self.test_labels = data['test_labels']
            self.logger.log_info(f"Dataset loaded successfully from {self.path}")
        except Exception as e:
            self.logger.log_error(e, "loading dataset from pickle")
            raise
    
    def _download_and_save(self) -> None:
        try:
            self.logger.log_info("Downloading CIFAR-10 dataset...")
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            self.train_images, self.train_labels = self._normalize_data(train_images, train_labels)
            self.test_images, self.test_labels = self._normalize_data(test_images, test_labels)
            
            with open(self.path, 'wb') as f:
                pickle.dump({
                    'train_images': self.train_images,
                    'train_labels': self.train_labels,
                    'test_images': self.test_images,
                    'test_labels': self.test_labels
                }, f)
            self.logger.log_info(f"Dataset downloaded and saved to {self.path}")
        except Exception as e:
            self.logger.log_error(e, "downloading and saving dataset")
            raise
