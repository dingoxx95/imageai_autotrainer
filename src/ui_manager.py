# src/ui_manager.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from typing import Optional
from . import default_logger
from .model_manager import ModelManager
from .gpu_config import GPUConfig
from .cifar_dataset import CifarDataset
from .training_orchestrator import TrainingOrchestrator

class TrainingUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CIFAR-10 Training Manager")

        # Inizializza componenti ML
        self.setup_ml_components()

        # Training control
        self.training_thread: Optional[threading.Thread] = None
        self.auto_training = False

        # UI Setup
        self.setup_ui()

        # Connettiamo i log alla UI
        default_logger.set_ui_log_callback(self.log_to_ui)

    def setup_ml_components(self):
        """Inizializza tutti i componenti ML necessari"""
        default_logger.log_info("Initializing ML components...")

        # Setup GPU
        self.gpu_config = GPUConfig()
        self.gpu_config.setup()

        # Prepare dataset
        self.dataset = CifarDataset()
        self.dataset.load()

        # Initialize model manager and orchestrator
        self.model_manager = ModelManager()
        self.orchestrator = TrainingOrchestrator(self.model_manager, self.dataset)

    def setup_ui(self):
        """Setup della UI"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control Panel
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Training Control", padding="5")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.start_button = ttk.Button(self.control_frame, text="Start Training", command=self.toggle_training)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1, padx=5, pady=5)

        # Performance Plot
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, pady=10)

        # Model Information
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Model Information", padding="5")
        self.info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        self.info_text = tk.Text(self.info_frame, height=10, width=70)
        self.info_text.grid(row=0, column=0, padx=5, pady=5)

        # Live Log Console
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Live Log", padding="5")
        self.log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

        self.log_text = tk.Text(self.log_frame, height=10, width=70)
        self.log_text.grid(row=0, column=0, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky='nsew')
        self.log_text['yscrollcommand'] = scrollbar.set

        # Aggiorna il display iniziale
        self.update_display()

    def log_to_ui(self, message: str):
        """Scrive i log nella UI in modo sicuro per i thread."""
        self.log_text.after(0, lambda: self._append_log_text(message))

    def _append_log_text(self, message: str):
        """Inserisce il messaggio nel widget di logging."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def toggle_training(self):
        """Avvia o interrompe il training"""
        if not self.auto_training:
            self.start_button.configure(text="Stop Training")
            self.auto_training = True
            self.training_thread = threading.Thread(target=self.run_training, daemon=True)
            self.training_thread.start()
        else:
            self.start_button.configure(text="Start Training")
            self.auto_training = False
            self.status_var.set("Status: Stopped")

    def run_training(self):
        """Esegue il training gestito da TrainingOrchestrator finché l'utente non preme Stop."""
        self.status_var.set("Status: Training in progress...")
        default_logger.log_info("Training started...")

        try:
            while self.auto_training:
                self.orchestrator.train_all_models()
                self.root.after(0, self.update_display)  # Aggiorna UI dopo ogni ciclo
        except Exception as e:
            default_logger.log_error(e, "training execution")
            self.status_var.set("Status: Error - Check logs")
        finally:
            self.start_button.configure(text="Start Training")
            self.auto_training = False

    def update_display(self):
        """Aggiorna il display con le informazioni più recenti"""
        try:
            # Aggiorna grafico performance
            self.ax.clear()
            history = self.model_manager.get_model_history()

            if history:
                accuracies = [m["accuracy"] for m in history]
                timestamps = [m["timestamp"] for m in history]

                self.ax.plot(range(len(accuracies)), accuracies, 'bo-', label='Accuracy')
                self.ax.set_title('Model Performance History')
                self.ax.set_xlabel('Training Iterations')
                self.ax.set_ylabel('Accuracy')
                self.ax.grid(True)

                # Aggiungi threshold line
                self.ax.axhline(y=self.model_manager.performance_threshold,
                                color='r', linestyle='--',
                                label=f'Threshold ({self.model_manager.performance_threshold})')

                self.ax.legend()
                self.figure.tight_layout()
                self.canvas.draw()

                # Aggiorna info modelli
                self.info_text.delete(1.0, tk.END)
                for i, model in enumerate(sorted(history, key=lambda x: x["accuracy"], reverse=True)):
                    info = f"Model {i+1}:\n"
                    info += f"  Name: {model['name']}\n"
                    info += f"  Accuracy: {model['accuracy']:.4f}\n"
                    info += f"  Date: {model['timestamp']}\n\n"
                    self.info_text.insert(tk.END, info)

        except Exception as e:
            default_logger.log_error(e, "updating display")

    def run(self):
        """Avvia la UI"""
        self.root.mainloop()
