'''
This python Script do a Profiling and a manual EDA

'''
import pandas as pd
import ydata_profiling
from ydata_profiling.utils.cache import cache_file
import utils as eda_utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import yeojohnson

#Define paths
data_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/data/"
save_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/intermediates/"
outcomes_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/outcomes/EDA/"

#Load data
df = pd.read_excel(data_path + "Caso Data Science Anlyst.xlsx",sheet_name=1)
print(df.head())
print(df.info())

################
# 0. Preprocessing
################
df = df.drop_duplicates().reset_index(drop=True) #drop duplicates
print(df.shape)
print(df.describe())
# Replace blanks with ' ' 
df.columns = df.columns.str.replace(' ', '_')
variables = ['Ship_Mode', 'Segment', 'City', 'State', 'Category']
for vars in variables:
    df[vars].replace(' ','_', regex = True, inplace = True)

df = df.dropna().reset_index(drop=True) #drop Nan's

df['City'] = df['City'].astype('category') #for the profiling EDA

for col in df.columns:
    if df[col].dtype=='object':
        print("Number of unique values in",col + ": ",df[col].nunique())

print(df['Sub-Category'].value_counts())


# %%
################
# 1. EDA
################

###
#Fast EDA with Profile Report (to view this results open the HTML file: "EDA_report.html" in a browser (e.g.Chrome))
###

report = df.profile_report(sort=None,  correlations={
    "pearson": {"calculate": True},
    "spearman": {"calculate": True},
    "kendall": {"calculate": True},
    "phi_k": {"calculate": True},
    "cramers": {"calculate": True}
}, html={'style':{'full_width':True}})
report.to_file(outcomes_path + "EDA_report.html")


fig_path = outcomes_path + 'general/'

variables = ['Sales', 'Quantity', 'Discount', 'Profit']

###################
# 2. Distribution of numerical variables
###################

# Box-plots
eda_utils.boxplots(df, variables, outcomes_path)

# Histograms
colors = ['blue', 'green', 'salmon', 'gold']
eda_utils.histograms(df, variables, colors, outcomes_path)


for col in ['Profit','Sales']:
    plt.figure(figsize=(15,8))
    ax = sns.barplot(x="State", y=col, data=df, palette="Set1")
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=15)
    plt.title(f"States VS {col}",fontsize=24)
    plt.xlabel("States",fontsize=20)
    plt.ylabel(col,fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/state_vs_{col}.png")
    plt.close()


for col in ['Category','Sub-Category', 'Region']:
    plt.figure(figsize=[12,8])
    ax = sns.barplot(x=col, y="Profit", data=df)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/{col}_vs_profit.png")



###################
# 3. Relations between variables
###################

# Corr matrix
eda_utils.corr_heatmap(df, variables, fig_path)


# Scatterplot
eda_utils.scatterplots(df, df.columns,'Category', fig_path)


# Sales and profit by State
state_data = df.groupby('State').agg({'Sales':'sum', 'Profit':'sum'}).sort_values(by='Sales', ascending=False)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

state_data['Sales'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Ventas por Estado')
axes[0].set_xlabel('Estado')
axes[0].set_ylabel('Ventas Totales')
axes[0].tick_params(axis='x', rotation=90)

state_data['Profit'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Ganancias por Estado')
axes[1].set_xlabel('Estado')
axes[1].set_ylabel('Ganancias Totales')
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(fig_path + '/state_sales_profit.png')

###################
# 4. Multi-level Analysis 
###################
mla_vars = ['Segment', 'Category', 'Sub-Category', 'Region', 'City', 'State']
N = 10
eda_utils.multi_level_analysis(df, mla_vars, N, fig_path)


###################
# 5. Removing Outliers
###################
print(df.shape)
df = eda_utils.remove_outliers(df,3.33)
print(df.shape)
df.to_csv(save_path + "cleaned_data.csv", index= False)