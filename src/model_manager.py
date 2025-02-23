# src/model_manager.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json
import os
from datetime import datetime

from matplotlib import pyplot as plt
from . import default_logger, MODEL_CONFIG
import shutil
from .ml_logger import MLLogger
from .cnn_model import CNNModel


@dataclass(slots=True)
class ModelManager:
    base_dir: str = "models"
    best_model_dir: str = "models/best"
    history_file: str = "models/history.json"
    logger: MLLogger = field(default_factory=lambda: default_logger)
    performance_threshold: float = 0.65  # Soglia minima di accuratezza
    max_stored_models: int = 5
    
    def __post_init__(self):
        print(f"Initializing ModelManager in {self.base_dir}")  # Debug print
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        if not os.path.exists(self.history_file):
            self._init_history()
    
    def _init_history(self) -> None:
        """Inizializza il file di storia dei modelli"""
        history = {
            "models": [],
            "best_model": None,
            "best_accuracy": 0.0
        }
        self._save_history(history)
    
    def get_model_history(self) -> List[Dict]:
        """Restituisce la storia delle performance dei modelli"""
        try:
            return self._load_history()["models"]
        except Exception as e:
            self.logger.log_error(e, "getting model history")
            return []
    
    def _load_history(self) -> Dict:
        """Carica la storia dei modelli"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.log_error(e, "loading model history")
            return self._init_history()
    
    def _save_history(self, history: Dict) -> None:
        """Salva la storia dei modelli"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.log_error(e, "saving model history")
    
    def _generate_model_name(self) -> str:
        """Genera un nome univoco per il modello"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"model_{timestamp}"
    
    def save_model(self, model: CNNModel, accuracy: float) -> str:
        """Salva un modello con il suo valore di accuratezza"""
        try:
            history = self._load_history()
            model_name = self._generate_model_name()
            model_path = os.path.join(self.base_dir, model_name)

            # Salva il modello
            model.model.save(model_path)

            # Registra il modello nella history
            model_info = {
                "name": model_name,
                "path": model_path,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
                "parameters": MODEL_CONFIG
            }

            history["models"].append(model_info)
            self._save_history(history)

            self._keep_best_models()
            return model_path

        except Exception as e:
            self.logger.log_error(e, "saving model")
            raise
        
    def _keep_best_models(self):
        """Mantiene solo i migliori 5 modelli salvati"""
        history = self._load_history()
        history["models"].sort(key=lambda x: x["accuracy"], reverse=True)

        while len(history["models"]) > self.max_stored_models:
            model_to_remove = history["models"].pop()
            if os.path.exists(model_to_remove["path"]):
                shutil.rmtree(model_to_remove["path"])
                self.logger.log_info(
                    f"Removed model {model_to_remove['name']} with accuracy {model_to_remove['accuracy']:.4f}"
                )

        if history["models"]:
            best_model = history["models"][0]
            history["best_model"] = best_model["name"]
            history["best_accuracy"] = best_model["accuracy"]

            best_path = os.path.join(self.best_model_dir, "best_model")
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(best_model["path"], best_path)

        self._save_history(history)
    
    def visualize_history(self) -> None:
        """Visualizza lo storico delle performance dei modelli"""
        try:
            history = self._load_history()
            models = history["models"]

            if not models:
                self.logger.log_warning("No models in history to visualize")
                return

            plt.figure(figsize=(12, 6))
            accuracies = [m["accuracy"] for m in models]
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in models]

            plt.plot(timestamps, accuracies, 'bo-', label='Model Accuracy')
            if history["best_accuracy"]:
                plt.axhline(y=history["best_accuracy"], color='r', linestyle='--', label=f'Best Accuracy ({history["best_accuracy"]:.4f})')

            plt.axhline(y=self.performance_threshold, color='g', linestyle=':', label=f'Threshold ({self.performance_threshold})')
            plt.title('Model Performance History')
            plt.xlabel('Training Time')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            plot_path = "models/performance_history.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            self.logger.log_info("Model performance history visualized.")

        except Exception as e:
            self.logger.log_error(e, "visualizing model history")