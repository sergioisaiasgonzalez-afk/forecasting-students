import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import linregress
import warnings



def graficar_serie(
    original,
    forecast,
    title
):
    fig=go.Figure() 
    fig.add_trace(
        go.Scatter(
            x=original.index,
            y=original.values,
            mode='lines',
            name='Valor Real',
            opacity=0.5
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='SES'
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo",
        yaxis_title="Valor"
    )
    fig.show()

def mape(
        original,
        forecast
):
    mask = forecast.notna()

    mape=np.mean(
        np.abs(
            (original[mask]-forecast[mask])/original[mask]
        )
    )*100

    return mape