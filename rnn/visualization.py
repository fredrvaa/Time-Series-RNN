import numpy as np
import matplotlib.pyplot as plt


def visualize_forecast(y: np.ndarray, y_hat: np.ndarray, forecast_start: int = 144):
    forecast_steps = y_hat.shape[0]
    plt.plot(x=np.arange(forecast_start),
             y=y[:forecast_start],
             label='History')

    plt.plot(x=np.arange(forecast_start, forecast_start+forecast_steps),
             y=y[forecast_start:forecast_start+forecast_steps],
             label='Target')

    plt.plot(x=np.arange(forecast_start, forecast_start+forecast_steps),
             y=y_hat,
             label='Forecast')

    plt.show()
