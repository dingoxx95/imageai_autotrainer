from . import default_logger, MODEL_CONFIG
from dataclasses import dataclass, field
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from .ml_logger import MLLogger
from .model_trainer import ModelTrainer

@dataclass(slots=True)
class Visualizer:
    trainer: ModelTrainer
    logger: MLLogger = field(default_factory=lambda: default_logger)
    class_names: Tuple[str, ...] = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )
    
    def plot_training_history(self) -> None:
        try:
            if not self.trainer.history:
                self.logger.log_warning("No training history available")
                return
                
            plt.figure(figsize=(12, 4))
            
            # Plot training & validation accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.trainer.history['accuracy'])
            plt.plot(self.trainer.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'])
            
            # Plot training & validation loss
            plt.subplot(1, 2, 2)
            plt.plot(self.trainer.history['loss'])
            plt.plot(self.trainer.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'])
            
            plt.tight_layout()
            plot_path = "training_history.png"
            plt.savefig(plot_path)
            self.logger.log_info(f"Training history plot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            self.logger.log_error(e, "plotting training history")
            raise
    
    def show_predictions(self, num_images: int = 5) -> None:
        try:
            # Get random test images
            indices = np.random.randint(0, len(self.trainer.dataset.test_images), num_images)
            test_images = self.trainer.dataset.test_images[indices]
            test_labels = self.trainer.dataset.test_labels[indices]
            
            predictions = self.trainer.predict_batch(test_images)
            
            plt.figure(figsize=(15, 3))
            for i in range(num_images):
                plt.subplot(1, num_images, i + 1)
                plt.imshow(test_images[i])
                true_label = self.class_names[np.argmax(test_labels[i])]
                pred_label = self.class_names[np.argmax(predictions[i])]
                color = 'green' if true_label == pred_label else 'red'
                plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
                plt.axis('off')
            
            plot_path = "predictions.png"
            plt.savefig(plot_path)
            self.logger.log_info(f"Predictions plot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            self.logger.log_error(e, "showing predictions")
            raise