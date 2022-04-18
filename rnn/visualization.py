from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def visualize_forecast(y: np.ndarray,
                       y_hat: np.ndarray,
                       ax: Optional[plt.axis] = None,
                       start_idx: int = 0,
                       sequence_length: int = 144,
                       history_length: int = 144):

    forecast_start_idx = start_idx + sequence_length
    forecast_steps = y_hat.shape[0]

    history = [np.arange(forecast_start_idx-history_length, forecast_start_idx),
               y[forecast_start_idx-history_length:forecast_start_idx]]

    target = [np.arange(forecast_start_idx-1, forecast_start_idx+forecast_steps),
             y[forecast_start_idx-1:forecast_start_idx+forecast_steps]]

    forecast = [np.arange(forecast_start_idx-1, forecast_start_idx+forecast_steps),
                np.insert(y_hat, 0, y[forecast_start_idx-1])]

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(*history, label='History')
        plt.plot(*target, label='Target')
        plt.plot(*forecast, label='Forecast')
        plt.legend()
        plt.show()
    else:
        ax.plot(*history, label='History')
        ax.plot(*target, label='Target')
        ax.plot(*forecast, label='Forecast')
