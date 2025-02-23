from . import default_logger, MODEL_CONFIG
from dataclasses import dataclass, field
from typing import Dict, Optional
import tensorflow as tf
import numpy as np
from datetime import datetime
from .ml_logger import MLLogger
from .cnn_model import CNNModel
from .cifar_dataset import CifarDataset
import json

@dataclass(slots=True)
class ModelTrainer:
    model: CNNModel
    dataset: CifarDataset
    logger: MLLogger = field(default_factory=lambda: default_logger)  # Questa era l'origine dell'errore
    batch_size: int = MODEL_CONFIG['BATCH_SIZE']  # Non serve field per valori semplici
    epochs: int = MODEL_CONFIG['EPOCHS']
    history: Optional[Dict] = None
    
    def train(self) -> None:
        try:
            self.logger.log_info(f"Starting training with batch_size={self.batch_size}, epochs={self.epochs}")
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # TensorBoard callback
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=f"./logs/tensorboard/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                histogram_freq=1
            )
            
            # Train the model
            self.history = self.model.model.fit(
                self.dataset.train_images,
                self.dataset.train_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.dataset.test_images, self.dataset.test_labels),
                callbacks=[
                    self.model.get_checkpoint_callback(),
                    early_stopping,
                    tensorboard_callback
                ]
            ).history
            
            for epoch, metrics in enumerate(zip(
                self.history['loss'], 
                self.history['accuracy'],
                self.history['val_loss'],
                self.history['val_accuracy']
            )):
                self.logger.log_training_step(epoch, {
                    'loss': metrics[0],
                    'accuracy': metrics[1],
                    'val_loss': metrics[2],
                    'val_accuracy': metrics[3]
                })
                
        except Exception as e:
            self.logger.log_error(e, "training model")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        try:
            self.logger.log_info("Evaluating model on test set")
            test_loss, test_accuracy = self.model.model.evaluate(
                self.dataset.test_images,
                self.dataset.test_labels,
                verbose=0
            )
            
            eval_metrics = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.logger.log_info(f"Evaluation metrics: {json.dumps(eval_metrics, indent=2)}")
            return eval_metrics
            
        except Exception as e:
            self.logger.log_error(e, "evaluating model")
            raise
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        try:
            predictions = self.model.model.predict(images)
            self.logger.log_info(f"Made predictions for {len(images)} images")
            return predictions
        except Exception as e:
            self.logger.log_error(e, "making predictions")
            raise