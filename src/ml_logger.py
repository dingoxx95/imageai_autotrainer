# ml_logger.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from typing import Optional, Dict, Any
import logging
import json
from datetime import datetime
import os
from pathlib import Path

class UIHandler(logging.Handler):
    """Handler di logging che inoltra i log alla UI tramite callback."""
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord):
        """Invia il log alla UI in modo thread-safe."""
        log_entry = self.format(record)
        if self.callback:
            try:
                self.callback(log_entry)
            except Exception:
                pass  # Evita crash se la UI non è attiva


@dataclass(slots=True)
class MLLogger:
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    model_name: str = "cifar10_cnn"
    file_handler: Optional[logging.FileHandler] = None
    console_handler: Optional[logging.StreamHandler] = None
    ui_handler: Optional[UIHandler] = None
    metrics_file: Optional[Path] = None
    _logger: Optional[logging.Logger] = None
    ui_log_callback: Optional[Callable[[str], None]] = None
    
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
    
    def set_ui_log_callback(self, callback: Callable[[str], None]):
        """Collega il logger alla UI per la visualizzazione live."""
        self.ui_log_callback = callback

        # Rimuovi l'eventuale handler UI precedente
        if self.ui_handler:
            self._logger.removeHandler(self.ui_handler)

        # Crea un nuovo handler per la UI
        self.ui_handler = UIHandler(callback)
        self.ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(self.ui_handler)
        
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
        """Logga metriche di training."""
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._logger.info(f"Epoch {epoch}: {metrics_str}")

        with open(self.metrics_file, 'a') as f:
            metrics['epoch'] = epoch
            metrics['timestamp'] = datetime.now().isoformat()
            json.dump(metrics, f)
            f.write('\n')

    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """Logga informazioni sul dataset"""
        self._logger.info(f"Dataset Info: {json.dumps(dataset_info, indent=2)}")

    def log_error(self, error: Exception, context: str = ""):
        """Logga un errore con contesto."""
        err_msg = f"[ERROR] {context}: {str(error)}"
        self._logger.error(err_msg, exc_info=True)

    def log_checkpoint(self, checkpoint_path: str) -> None:
        """Logga salvataggio checkpoint"""
        self._logger.info(f"Checkpoint saved: {checkpoint_path}")

    def close(self) -> None:
        """Chiude i file handler."""
        if self.file_handler:
            self.file_handler.close()
            self._logger.removeHandler(self.file_handler)
        if self.console_handler:
            self.console_handler.close()
            self._logger.removeHandler(self.console_handler)
        if self.ui_handler:
            self._logger.removeHandler(self.ui_handler)