import matplotlib.pyplot as plt
from utils_json import *
from loguru import logger


def plot_performance_metrics(input_json, output_png):
    perf_data = load_json(input_json)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    # axes = plt.gca()
    ax.set_ylim([0.16, 0.30])
    ax.plot(perf_data["epoch"], perf_data['train_loss'], label='Train')
    ax.plot(perf_data["epoch"], perf_data['val_loss'], label='Validation')
    ax.set_xticks(perf_data["epoch"])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(output_png)
    plt.close()
    logger.info(f"Save performance plot {output_png}")


