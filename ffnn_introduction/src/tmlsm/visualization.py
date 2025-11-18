"""Visualization utilities for training."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from jaxtyping import Array
import klax


class AnimationCallback(klax.Callback):
    """Callback für Animation während klax.fit()."""
    
    def __init__(
        self,
        true_y: Array,  # <-- Geändert: nimm y-Werte direkt
        x_train: Array,
        y_train: Array,
        x_plot: Array,
        save_every: int = 50,
        filename: str = "training.gif"
    ):
        self.true_y = true_y
        self.x_train = x_train
        self.y_train = y_train
        self.x_plot = x_plot
        self.save_every = save_every
        self.filename = filename
        self.frames = []
    
    def __call__(self, cbargs):
        """Wird von klax.fit() aufgerufen mit CallbackArgs."""
        step = cbargs.step
        model = cbargs.model
        loss = cbargs.loss
        
        if step % self.save_every == 0:
            # Finalize model für Predictions
            model_finalized = klax.finalize(model)
            y_pred = jax.vmap(model_finalized)(self.x_plot)
            
            self.frames.append({
                'step': step,
                'loss': float(loss),
                'x': jnp.array(self.x_plot),
                'y_pred': jnp.array(y_pred),
                'y_true': jnp.array(self.true_y),  # <-- Verwende direkt die y-Werte
            })
        
        return False  # Return False um Training fortzusetzen
    
    def save_animation(self):
        """Speichert die Animation als GIF."""
        if not self.frames:
            print("Keine Frames vorhanden!")
            return
        
        print(f"Erstelle Animation mit {len(self.frames)} Frames...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        losses = [f['loss'] for f in self.frames]
        steps_list = [f['step'] for f in self.frames]
        
        def animate(i):
            ax1.clear()
            ax2.clear()
            
            frame = self.frames[i]
            
            # Plot 1: Funktionsapproximation
            ax1.plot(frame['x'], frame['y_true'], 'g-', linewidth=2, 
                    label='Bathtub Function', alpha=0.7)
            ax1.plot(frame['x'], frame['y_pred'], 'b-', linewidth=2, 
                    label='Model')
            ax1.scatter(self.x_train[::10], self.y_train[::10], c='red', s=30, 
                       alpha=0.5, label='Training Data')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(f'Step {frame["step"]}, Loss: {frame["loss"]:.6f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss Kurve
            ax2.plot(steps_list[:i+1], losses[:i+1], 'r-', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, animate, frames=len(self.frames), 
                           interval=100, repeat=True)
        writer = PillowWriter(fps=10)
        anim.save(self.filename, writer=writer)
        plt.close()
        print(f"✓ Animation gespeichert: {self.filename}")