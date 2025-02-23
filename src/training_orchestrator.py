# src/training_orchestrator.py
from dataclasses import dataclass
from . import default_logger
from .model_manager import ModelManager
from .model_trainer import ModelTrainer
from .cnn_model import CNNModel
from .cifar_dataset import CifarDataset

@dataclass
class TrainingOrchestrator:
    model_manager: ModelManager
    dataset: CifarDataset

    def train_all_models(self):
        """Carica tutti i modelli salvati, li addestra e salva solo i migliori 5."""
        history = self.model_manager.get_model_history()
        trained_models = []

        if not history:
            default_logger.log_info("No saved models found. Training a new model from scratch.")
            cnn = CNNModel()
            cnn.build()
            models_to_train = [cnn]
        else:
            models_to_train = []
            for model_data in history:
                model_path = model_data["path"]
                default_logger.log_info(f"Loading model: {model_path}")

                cnn = CNNModel()
                cnn.load(model_path)
                models_to_train.append(cnn)

        for cnn in models_to_train:
            trainer = ModelTrainer(model=cnn, dataset=self.dataset)
            trainer.train()
            eval_metrics = trainer.evaluate()
            trained_models.append((cnn, eval_metrics["test_accuracy"]))

        # Salviamo solo i migliori 5 modelli
        for cnn, accuracy in trained_models:
            self.model_manager.save_model(cnn, accuracy)

        self.model_manager._keep_best_models()
        default_logger.log_info("Training complete. Top 5 models saved.")
