import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    def __init__(self, df):
        self.df = df
        self.X = df['x'].values.reshape(-1, 1)
        self.y = df['y'].values
        self.slope = None
        self.intercept = 0

    def cal_sr(self, slope, intercept):
        y_pred = slope * self.df['x'] + intercept
        SR = (self.df['y'] - y_pred) ** 2
        return SR

    def SSR_slope(self, min_slope, max_slope, number):
        slopes = np.linspace(min_slope, max_slope, number)
        SSR_graph = {}

        for slope in slopes:
            SR = self.cal_sr(slope, self.intercept)
            SSR = SR.sum()
            SSR_graph[slope] = SSR

        SSR_df = pd.DataFrame(list(SSR_graph.items()), columns=['Eğim', 'SSR'])
        return slopes, SSR_df

    def scatter_plot(self, data, x, y,  x_label, y_label, title):
        fig, ax = plt.subplots()
        ax.scatter(data[x], data[y], color='blue')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        st.pyplot(fig)

    def scatter_plot_ssr(self, slope, plot=False):
        X_mean = np.mean(self.X)
        y_mean = np.mean(self.y)
        x_line = np.linspace(self.X.min(), self.X.max(), 100)
        intercept = y_mean - slope * X_mean
        y_line = slope * x_line + intercept

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(self.df['x'], self.df['y'], color='blue')
            ax.plot(x_line, y_line, color='red', label=f'Line: y={slope}x + {intercept:.2f}')

            for i in range(len(self.df)):
                x_val = self.df['x'].iloc[i]
                y_val = self.df['y'].iloc[i]
                y_line_val = slope * x_val + intercept
                ax.plot([x_val, x_val], [y_val, y_line_val], color='blue', linestyle='--', linewidth=0.8)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Dataset Scatter Plot')
            plt.legend()
            st.pyplot(fig)
        return intercept

    def fit(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        self.slope = model.coef_[0]
        self.intercept = model.intercept_
        return self.slope, self.intercept

    def gradient_descent(self, learning_rate, iterations):
        slope, intercept = 0, 0
        n = len(self.y)
        cost_list = []

        for _ in range(iterations):
            slope_gradient = 0
            intercept_gradient = 0
            cost = 0
            for i in range(n):
                x = self.X[i, 0]
                y_pred = slope * x + intercept
                error = self.y[i] - y_pred
                cost += error ** 2
                slope_gradient += -(2 / n) * x * error
                intercept_gradient += -(2 / n) * error

            cost_new = cost / (2 * n)
            cost_list.append(cost_new)
            if cost_new <= 1e-6:
                break
            slope -= learning_rate * slope_gradient
            intercept -= learning_rate * intercept_gradient

        self.slope, self.intercept = slope, intercept
        return slope, intercept, cost_list

    def plot_cost(self, cost_list):
        plt.figure(figsize=(10, 6))
        plt.title('Cost Function')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.plot(cost_list)
        st.pyplot(plt)

    def r2_cal(self, slope, intercept):
        sum_square = 0
        sum_residual = 0
        y_mean = np.mean(self.y)
        for i in range(len(self.X)):
            y_pred = slope * self.X[i] + intercept
            sum_residual += (self.y[i] - y_pred) ** 2
            sum_square += (self.y[i] - y_mean) ** 2
        score = 1 - (sum_residual / sum_square)
        return score

    def mse_cal(self, slope, intercept):
        sum_residual = 0
        n = len(self.y)
        for i in range(n):
            y_pred = slope * self.X[i] + intercept
            sum_residual += (self.y[i] - y_pred) ** 2
        mse = sum_residual / n
        return mse