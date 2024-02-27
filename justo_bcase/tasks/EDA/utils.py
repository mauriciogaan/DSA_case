'''
This Python script contains necessary functions for the EDA

'''


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def histograms(data, variables, colors, base_path):
    file_paths = []
    for var, color in zip(variables, colors):
        # Creating the figure
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.histplot(data[var], kde=True, color=color)
        plt.title(f'{var} Distribution', fontsize=15)
        plt.xlabel(var, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Save
        file_path = f'{base_path}histograms/{var.lower()}.png'
        plt.savefig(file_path)
        plt.close() 
        file_paths.append(file_path)
    return file_paths

def boxplots(data, variables, base_path):
    file_paths_boxplots = []
    for var in variables:
        # Creating the figure
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data[var], color='green')
        plt.title(f'{var} Boxplot', fontsize=15)
        plt.xlabel(var, fontsize=12)
        
        # Save
        file_path = f'{base_path}boxplots/{var.lower()}_boxplot.png'
        plt.savefig(file_path)
        plt.close()  
        file_paths_boxplots.append(file_path)
    return file_paths_boxplots


def corr_heatmap(data, variables, base_path):
    # heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[variables].corr(),cmap='Reds',annot=True);
    plt.rcParams['figure.figsize']=(10,5)
    plt.title('Heatmap of Correlation Matrix', fontsize=15)

    #Save
    plt.savefig(base_path + 'heatmap.png')
    plt.close() 
    

def scatterplots(data, variables, group, base_path):
    
    # Creating the figure 
    sns.set(style="ticks")
    scatter_matrix = sns.pairplot(data[variables], hue=group)
    scatter_matrix.fig.suptitle('Scatter Plot Matrix', y=1.02)

    # Save
    scatter_matrix.savefig(base_path + 'scatterplot.png')
    plt.close() 


def multi_level_analysis(data, variables,N, base_path):
    multi_level_analysis = data.groupby(variables).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()

    # Profit Margin
    multi_level_analysis['Profit Margin'] = multi_level_analysis['Profit'] / multi_level_analysis['Sales'] * 100

    # Find extreme Values
    multi_level_analysis = multi_level_analysis.sort_values('Profit Margin', ascending=False)
    top_sales = multi_level_analysis.nlargest(N, 'Profit Margin')
    lowest_profits = multi_level_analysis.nsmallest(N, 'Profit Margin')
    interesting_values = pd.concat([top_sales, lowest_profits]).drop_duplicates().reset_index(drop=True)

    # store results
    interesting_values.to_csv(base_path + 'multi_level_analysis.csv', index=False)
    

def remove_outliers(dataset, k=3.33):
    # Initialize a DataFrame to mark rows to keep
    indices_to_keep = pd.Series([True] * len(dataset))
    
    for col in dataset.columns:
        if dataset[col].dtype in ["int64", "float64"]:
            mean = dataset[col].mean()
            std = dataset[col].std()
            # Create a mask to identify values within the acceptable range
            is_outlier = (dataset[col] < (mean - k * std)) | (dataset[col] > (mean + k * std))
            # Update the indices to keep by excluding the current outliers
            indices_to_keep &= ~is_outlier
            
    # Filter the original dataset to only include rows without outliers
    dataset_without_outliers = dataset[indices_to_keep]
    return dataset_without_outliers