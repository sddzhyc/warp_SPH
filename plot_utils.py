import matplotlib.pyplot as plt
import csv
import os

def save_grid_search_to_csv(vy_values, loss_list, grad_list, file_path):
    """
    Save grid search results to a CSV file.
    """
    try:
        # verify directory exists, if file_path has a directory component
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vy', 'Loss', 'Gradient_Y'])
            for vy, loss, grad in zip(vy_values, loss_list, grad_list):
                writer.writerow([vy, loss, grad])
        print(f"Grid search results saved to {file_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

def plot_grid_search_results(vy_values, loss_list, grad_list, vy_min, vy_max, writer):
    """
    Generate summary plot for grid search and log to TensorBoard.
    """
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Initial Velocity Y (vy)')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(vy_values, loss_list, color=color, marker='o', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Gradient Y', color=color)
        ax2.plot(vy_values, grad_list, color=color, linestyle='--', marker='x', label='Gradient Y')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Loss and Gradient Layout over Vy Range [{vy_min}, {vy_max}]")
        fig.tight_layout()
        
        writer.add_figure('GridSearch/Overview', fig, 0)
        plt.close(fig)
        print("Overview plot added to TensorBoard.")
    except Exception as e:
        print(f"Failed to generate plot: {e}")
