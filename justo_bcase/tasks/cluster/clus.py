'''
This Python script performs an K-Means clustering (Using Elbow Method to set # of clusters)
and creates some figures and stats to learn more about the clusters.

It also uses the Local Outlier Factor (LOF) algorithm for anomaly detection and 
check if there are patterns between clusters

'''



from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

# Define paths
data_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/intermediates/"
outcomes_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/outcomes/cluster/"


#######################
# 1. Preprocessing
#######################

# Load and preprocess data
df = pd.read_csv(data_path + "cleaned_data.csv")
df_k = df[['Ship_Mode', 'Quantity', 'Sales', 'Profit', 'Sub-Category', 'Category', 'Region']].copy()

categorical_features = ['Ship_Mode', 'Sub-Category', 'Category', 'Region']
numerical_features = ['Quantity', 'Sales', 'Profit']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Standardize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ] 
)

df_k_encoded_array = preprocessor.fit_transform(df_k)
df_k_encoded_dense = df_k_encoded_array.toarray() if hasattr(df_k_encoded_array, "toarray") else df_k_encoded_array
new_columns = preprocessor.get_feature_names_out()

print(df_k_encoded_dense.shape) 
print(len(new_columns))  

# Create the DataFrame 
df_k_encoded = pd.DataFrame(df_k_encoded_dense, columns=new_columns)

#######################
# 2. Elbow Method
#######################

# Use the Elbow Method to find the optimal number of clusters
k = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_k_encoded)
    k.append(kmeans.inertia_)

# Plotting the results 
plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), k, marker='o')
plt.title('Elbow Method results')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of intra-cluster squares')
plt.show()

#######################
# 3. K-Means
#######################


# Fit MiniBatchKMeans for efficient clustering
kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=10)
y_pred = kmeans.fit_predict(df_k_encoded)

# Add a cluster column
df['k_means_clusters'] = pd.Series(y_pred)


#######################
# 4. Stats and Vizualizations
#######################

# Generate descriptive statistics by cluster
descriptive_stats = df.groupby('k_means_clusters')[['Sales', 'Quantity', 'Discount', 'Profit']].describe()
descriptive_stats.to_csv(outcomes_path + 'descriptive_stats.csv')

# Plotting scatter and 3D scatter plots to visualize clusters based on different features
fig = px.scatter(data_frame=df, x='Sales', y='Quantity', color='k_means_clusters')
fig.write_html(outcomes_path + "/Sales_v_Quant.html")
fig.show()

fig = px.scatter(data_frame=df, x='Profit', y='Sales', color='k_means_clusters')
fig.write_html(outcomes_path + "/Sales_v_Profit.html")
fig.show()

fig = px.scatter_3d(data_frame=df, x='Sales', y='Profit', z='Region', color='k_means_clusters')
fig.write_html(outcomes_path + "/Sales_v_Profit_v_Region.html")
fig.show()

fig = px.scatter_3d(data_frame=df, x='Sales', y='Profit', z='Sub-Category', color='k_means_clusters')
fig.write_html(outcomes_path + "/Sales_v_Profit_v_Sub-cat.html")
fig.show()

fig = px.scatter_3d(data_frame=df, x='Sales', y='Quantity', z='Ship_Mode', color='k_means_clusters')
fig.write_html(outcomes_path + "/Sales_v_Profit_v_Ship.html")
fig.show()

# Box plots for Sales, Quantity, and Profit by cluster 
fig, axs = plt.subplots(3, 1, figsize=(10, 18))


sns.boxplot(x='k_means_clusters', y='Sales', data=df, ax=axs[0])
axs[0].set_title('Sales Distribution by Cluster')
axs[0].set_yscale('log')  # Using log scale for better visibility of data spread

sns.boxplot(x='k_means_clusters', y='Quantity', data=df, ax=axs[1])
axs[1].set_title('Quantity Distribution by Cluster')

sns.boxplot(x='k_means_clusters', y='Profit', data=df, ax=axs[2])
axs[2].set_title('Profit Distribution by Cluster')
axs[2].set_yscale('symlog')  # Symmetric log scale for positive and negative values
plt.savefig(outcomes_path + 'boxplot_clusters.png')


#######################
# 5. Local Outlier Factor (LOF)
#######################

# Initialize the Local Outlier Factor (LOF) algorithm for anomaly detection
lof = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=0.05)

# (-1 for outliers, 1 for inliers)
numeric_columns = ['Sales', 'Quantity', 'Discount', 'Profit']
data_numeric = df[numeric_columns]
outliers = lof.fit_predict(data_numeric)
df['LOF_Outlier'] = outliers

outliers_distribution = df['LOF_Outlier'].value_counts()
print(outliers_distribution)

outliers_data = df[df['LOF_Outlier'] == -1]
inliers_data = df[df['LOF_Outlier'] == 1]
outliers_count_per_cluster = outliers_data['k_means_clusters'].value_counts().sort_index()
inliers_count_per_cluster = inliers_data['k_means_clusters'].value_counts().sort_index()

# Calculate the percentage of outliers in each cluster
comparison_df = pd.DataFrame({
    'Outliers': outliers_count_per_cluster,
    'Inliers': inliers_count_per_cluster
}).fillna(0)  # Ensure there are no NaNs

# Calculate the percentage of outliers in each cluster
comparison_df['Outlier_Percentage'] = (comparison_df['Outliers'] / (comparison_df['Outliers'] + comparison_df['Inliers'])) * 100
print(comparison_df)

# Plot
comparison_df[['Outliers','Inliers']].plot(kind='bar', figsize=(14, 7), title="Outliers and Inliers Distribution by Cluster")
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig(outcomes_path + 'outliers_fig.png')

# Save the DataFrame 
df.to_csv(outcomes_path + "clusters.csv")
