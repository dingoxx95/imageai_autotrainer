from . import default_logger, MODEL_CONFIG
from dataclasses import dataclass, field
import tensorflow as tf
from .ml_logger import MLLogger

@dataclass(slots=True)
class GPUConfig:
    gpus: list = field(default_factory=list)
    logger: MLLogger = field(default_factory=lambda: default_logger)
    
    def setup(self) -> None:
        self.gpus = tf.config.experimental.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU set to dynamic memory growth")
            except RuntimeError as e:
                print(e)
    
    def print_info(self) -> None:
        gpu_info = {
            "num_gpus": len(tf.config.list_physical_devices('GPU')),
            "cuda_available": tf.test.is_built_with_cuda(),
            "compute_device": int(tf.keras.backend.get_value(tf.keras.backend.learning_phase()))
        }
        self.logger.log_gpu_info(gpu_info)