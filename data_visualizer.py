# data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataVisualizer:
    @staticmethod
    def plot_category_distribution(data, category):
        sns.catplot(x=category, data=data, kind='count')
        plt.show()

    @staticmethod
    def plot_feature_vs_target(data, feature, target):
        plt.figure(figsize=(5, 5))
        sns.barplot(x=target, y=feature, data=data)
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(data):
        numeric_data = data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
        plt.show()
