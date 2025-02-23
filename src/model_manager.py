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
        try:
            history = self._load_history()
            model_name = self._generate_model_name()
            model_path = os.path.join(self.base_dir, model_name)
            
            # Salva il nuovo modello
            model.model.save(model_path)
            
            # Prepara info del nuovo modello
            model_info = {
                "name": model_name,
                "path": model_path,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
                "parameters": MODEL_CONFIG
            }
            
            # Aggiungi il nuovo modello alla storia
            history["models"].append(model_info)
            
            # Ordina i modelli per accuracy (dal migliore al peggiore)
            history["models"].sort(key=lambda x: x["accuracy"], reverse=True)
            
            # Se abbiamo più di max_stored_models, rimuovi i peggiori
            while len(history["models"]) > self.max_stored_models:
                model_to_remove = history["models"].pop()  # Rimuove il peggiore
                if os.path.exists(model_to_remove["path"]):
                    shutil.rmtree(model_to_remove["path"])
                    self.logger.log_info(
                        f"Removed model {model_to_remove['name']} "
                        f"with accuracy {model_to_remove['accuracy']:.4f}"
                    )
            
            # Aggiorna best model se necessario
            if accuracy > history.get("best_accuracy", 0):
                self.logger.log_info(
                    f"New best model! "
                    f"Accuracy improved from {history.get('best_accuracy', 0):.4f} to {accuracy:.4f}"
                )
                best_path = os.path.join(self.best_model_dir, "best_model")
                if os.path.exists(best_path):
                    shutil.rmtree(best_path)
                shutil.copytree(model_path, best_path)
                history["best_model"] = model_name
                history["best_accuracy"] = accuracy
            
            self._save_history(history)
            return model_path
            
        except Exception as e:
            self.logger.log_error(e, "saving model")
            raise
    
    def load_latest_model(self) -> Optional[str]:
        """Carica il modello più recente se supera la soglia di performance"""
        try:
            history = self._load_history()
            if not history["models"]:
                return None
            
            latest_model = history["models"][-1]
            if latest_model["accuracy"] >= self.performance_threshold:
                return latest_model["path"]
            else:
                self.logger.log_warning(
                    f"Latest model accuracy ({latest_model['accuracy']:.4f}) "
                    f"below threshold ({self.performance_threshold})"
                )
                return None
                
        except Exception as e:
            self.logger.log_error(e, "loading latest model")
            return None
    
    def load_best_model(self) -> Optional[str]:
        """Carica il miglior modello"""
        try:
            history = self._load_history()
            if history["best_model"]:
                return os.path.join(self.best_model_dir, "best_model")
            return None
        except Exception as e:
            self.logger.log_error(e, "loading best model")
            return None
    
    def get_model_history(self) -> List[Dict]:
        """Restituisce la storia delle performance dei modelli"""
        try:
            history = self._load_history()
            return history["models"]
        except Exception as e:
            self.logger.log_error(e, "getting model history")
            return []
        
    def visualize_history(self) -> None:
        """Visualizza lo storico delle performance dei modelli"""
        try:
            history = self._load_history()
            models = history["models"]
            
            if not models:
                self.logger.log_warning("No models in history to visualize")
                return
            
            # Prepara i dati per il plot
            plt.figure(figsize=(12, 6))
            accuracies = [m["accuracy"] for m in models]
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in models]
            
            # Plot principale
            plt.plot(timestamps, accuracies, 'bo-', label='Model Accuracy')
            
            # Aggiungi linea per best accuracy
            if history["best_accuracy"]:
                plt.axhline(y=history["best_accuracy"], color='r', linestyle='--', 
                        label=f'Best Accuracy ({history["best_accuracy"]:.4f})')
            
            # Aggiungi linea soglia
            plt.axhline(y=self.performance_threshold, color='g', linestyle=':',
                    label=f'Threshold ({self.performance_threshold})')
            
            plt.title('Model Performance History')
            plt.xlabel('Training Time')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Salva il plot
            plot_path = "models/performance_history.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            # Log testuale delle performance
            self.logger.log_info("\nModel Performance History:")
            for model in sorted(models, key=lambda x: datetime.fromisoformat(x["timestamp"])):
                self.logger.log_info(
                    f"Model: {model['name']}, "
                    f"Accuracy: {model['accuracy']:.4f}, "
                    f"Date: {datetime.fromisoformat(model['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"
                )
            self.logger.log_info(f"\nBest model accuracy: {history['best_accuracy']:.4f}")
            
        except Exception as e:
            self.logger.log_error(e, "visualizing model history")