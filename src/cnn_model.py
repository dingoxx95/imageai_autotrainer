from . import default_logger, MODEL_CONFIG
from dataclasses import dataclass, field
from typing import Tuple, Optional
import tensorflow as tf
from keras import layers, models
import os
from .ml_logger import MLLogger

@dataclass(slots=True)
class CNNModel:
    input_shape: Tuple[int, int, int] = MODEL_CONFIG['INPUT_SHAPE']
    num_classes: int = MODEL_CONFIG['NUM_CLASSES']
    model: Optional[models.Sequential] = None
    checkpoint_path: str = "training_checkpoints/cp-{epoch:04d}.ckpt"
    logger: MLLogger = field(default_factory=lambda: default_logger)
    current_model_path: Optional[str] = None  # Path dinamico gestito da ModelManager
    
    def build(self) -> None:
        try:
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model_info = {
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "optimizer": "adam",
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"]
            }
            self.logger.log_model_summary(model_info)
            
        except Exception as e:
            self.logger.log_error(e, "building model")
            raise
    
    def save(self, path: Optional[str] = None) -> None:
        if self.model:
            try:
                save_path = path or self.current_model_path
                if save_path:
                    self.model.save(save_path)
                    self.current_model_path = save_path
                    self.logger.log_info(f"Model saved to {save_path}")
                else:
                    self.logger.log_warning("No path provided to save model")
            except Exception as e:
                self.logger.log_error(e, "saving model")
                raise
    
    def load(self, path: Optional[str] = None) -> None:
        try:
            load_path = path or self.current_model_path
            if load_path and os.path.exists(load_path):
                self.model = tf.keras.models.load_model(load_path)
                self.current_model_path = load_path
                self.logger.log_info(f"Model loaded from {load_path}")
            else:
                self.logger.log_warning(f"Model file not found at {load_path}")
        except Exception as e:
            self.logger.log_error(e, "loading model")
            raise
        
    def summary(self) -> None:
        if self.model:
            # Cattura l'output del summary in una stringa
            from io import StringIO
            summary_str = StringIO()
            self.model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
            self.logger.log_info(f"Model Summary:\n{summary_str.getvalue()}")
            
    def get_checkpoint_callback(self) -> tf.keras.callbacks.ModelCheckpoint:
        callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            save_weights_only=True,
            period=5
        )
        self.logger.log_info(f"Checkpoint callback created: {self.checkpoint_path}")
        return callback