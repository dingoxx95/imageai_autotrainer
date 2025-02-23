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
from .cnn_model import CNNModel
from .model_trainer import ModelTrainer
from .visualizer import Visualizer

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
        
    def setup_ml_components(self):
        """Inizializza tutti i componenti ML necessari"""
        default_logger.log_info("Initializing ML components...")
        
        # Setup GPU
        self.gpu_config = GPUConfig()
        self.gpu_config.setup()
        
        # Prepare dataset
        self.dataset = CifarDataset()
        self.dataset.load()
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
    def setup_ui(self):
        """Setup della UI"""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Training Control", padding="5")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Start Auto Training", 
            command=self.toggle_training
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Performance plot
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, pady=10)
        
        # Model info
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Model Information", padding="5")
        self.info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.info_text = tk.Text(self.info_frame, height=10, width=70)
        self.info_text.grid(row=0, column=0, padx=5, pady=5)
        
        # Aggiorna display iniziale
        self.update_display()
        
    def toggle_training(self):
        """Attiva/disattiva il training automatico"""
        self.auto_training = not self.auto_training
        if self.auto_training:
            self.start_button.configure(text="Stop Auto Training")
            self.training_thread = threading.Thread(target=self.auto_train)
            self.training_thread.daemon = True  # Il thread si fermerà quando chiudiamo la UI
            self.training_thread.start()
        else:
            self.start_button.configure(text="Start Auto Training")
            self.status_var.set("Status: Stopped")
    
    def auto_train(self):
        """Esegue il training in loop"""
        while self.auto_training:
            try:
                self.status_var.set("Status: Training in progress...")
                
                # Create and train model
                cnn = CNNModel()
                latest_model_path = self.model_manager.load_latest_model()
                
                if latest_model_path:
                    cnn.load(latest_model_path)
                    default_logger.log_info(f"Continuing training from model: {latest_model_path}")
                else:
                    cnn.build()
                    default_logger.log_info("Starting training from scratch")
                
                # Train
                trainer = ModelTrainer(model=cnn, dataset=self.dataset)
                trainer.train()
                
                # Evaluate
                eval_metrics = trainer.evaluate()
                
                # Save and update display
                self.model_manager.save_model(cnn, eval_metrics['test_accuracy'])
                self.root.after(0, self.update_display)
                
                self.status_var.set(f"Status: Last accuracy: {eval_metrics['test_accuracy']:.4f}")
                
            except Exception as e:
                default_logger.log_error(e, "auto training")
                self.status_var.set(f"Status: Error - {str(e)}")
                self.auto_training = False
                self.start_button.configure(text="Start Auto Training")
    
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