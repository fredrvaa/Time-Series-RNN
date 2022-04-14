import numpy as np
import matplotlib.pyplot as plt


def visualize_forecast(y: np.ndarray, y_hat: np.ndarray, history_length: int = 144, forecast_start: int = 144):
    forecast_steps = y_hat.shape[0]
    plt.plot(np.arange(forecast_start-history_length, forecast_start),
             y[forecast_start-history_length:forecast_start],
             label='History')

    plt.plot(np.arange(forecast_start-1, forecast_start+forecast_steps),
             y[forecast_start-1:forecast_start+forecast_steps],
             label='Target')

    plt.plot(np.arange(forecast_start-1, forecast_start+forecast_steps),
             np.insert(y_hat, 0, y[forecast_start-1]),
             label='Forecast')
    
    plt.legend()
    plt.show()
