import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

class DataVisualizer:
    def __init__(self, save_dir='static/plots'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_category_distribution(self, data, column, filename):
        plt.figure(figsize=(10, 6))
        data[column].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_feature_vs_target(self, data, feature, target, filename):
        plt.figure(figsize=(10, 6))
        plt.scatter(data[feature], data[target], alpha=0.5)
        plt.title(f'{feature} vs {target}')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_heatmap(self, data, feature, target, filename):
        print(f"Feature: {feature}, Target: {target}")
        print(f"Available columns in data: {data.columns.tolist()}")

        # Adjust figure size for better readability
        plt.figure(figsize=(20, 12))  # Larger width and height for better visibility

        # Create the cross-tabulation
        cross_tab = pd.crosstab(data[feature], data[target])

        # Increase font size for annotations and labels
        sns.heatmap(
            cross_tab,
            annot=True,
            fmt='d',
            cmap='Blues',
            annot_kws={"size": 10},  # Font size for annotations
            cbar_kws={"shrink": 0.75}  # Adjust color bar size
        )

        plt.title(f'{feature} vs {target}', fontsize=18)  # Larger font size for title
        plt.xlabel(target, fontsize=14)  # Larger font size for X-axis label
        plt.ylabel(feature, fontsize=14)  # Larger font size for Y-axis label

        # Save the figure
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight')  # Ensures nothing is cropped
        plt.close()


    # @staticmethod
    # def plot_category_distribution(data, category):
    #     sns.catplot(x=category, data=data, kind='count')
    #     plt.show()
    #
    # @staticmethod
    # def plot_feature_vs_target(data, feature, target):
    #     plt.figure(figsize=(5, 5))
    #     sns.barplot(x=target, y=feature, data=data)
    #     plt.show()

    @staticmethod
    def plot_correlation_heatmap(data):
        numeric_data = data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
        plt.show()