# ml_logger.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import json
from datetime import datetime
import os
from pathlib import Path

@dataclass(slots=True)
class MLLogger:
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    model_name: str = "cifar10_cnn"
    file_handler: Optional[logging.FileHandler] = None
    console_handler: Optional[logging.StreamHandler] = None
    metrics_file: Optional[Path] = None
    _logger: Optional[logging.Logger] = None
    
    def __post_init__(self):
        # Crea directory dei log se non esiste
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup del logger principale
        self._logger = logging.getLogger(self.model_name)
        self._logger.setLevel(logging.INFO)
        
        # Formattazione timestamp
        fmt = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler per file
        log_file = self.log_dir / f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setFormatter(fmt)
        self._logger.addHandler(self.file_handler)
        
        # Handler per console
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(fmt)
        self._logger.addHandler(self.console_handler)
        
        # File per metriche
        self.metrics_file = self.log_dir / f"{self.model_name}_metrics.jsonl"
    
    def log_info(self, message: str) -> None:
        """Logga messaggi informativi"""
        self._logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Logga messaggi di warning"""
        self._logger.warning(message)
    
    def log_gpu_info(self, gpu_info: Dict[str, Any]) -> None:
        """Logga informazioni sulla GPU"""
        self._logger.info(f"GPU Configuration: {json.dumps(gpu_info, indent=2)}")
    
    def log_model_summary(self, model_info: Dict[str, Any]) -> None:
        """Logga il summary del modello"""
        self._logger.info(f"Model Architecture:\n{json.dumps(model_info, indent=2)}")
    
    def log_training_step(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Logga metriche di training per ogni epoca"""
        # Log testuale
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._logger.info(f"Epoch {epoch}: {metrics_str}")
        
        # Log strutturato per analisi
        with open(self.metrics_file, 'a') as f:
            metrics['epoch'] = epoch
            metrics['timestamp'] = datetime.now().isoformat()
            json.dump(metrics, f)
            f.write('\n')
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """Logga informazioni sul dataset"""
        self._logger.info(f"Dataset Info: {json.dumps(dataset_info, indent=2)}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Logga errori con contesto"""
        self._logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_checkpoint(self, checkpoint_path: str) -> None:
        """Logga salvataggio checkpoint"""
        self._logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def close(self) -> None:
        """Chiude i file handler"""
        if self.file_handler:
            self.file_handler.close()
            self._logger.removeHandler(self.file_handler)
        if self.console_handler:
            self.console_handler.close()
            self._logger.removeHandler(self.console_handler)