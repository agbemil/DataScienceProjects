

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.feature_selection import SelectKBest, f_classif
from scipy.cluster.hierarchy import linkage
from sklearn.utils import resample
#import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
#import numpy as np
#import networkx as nx
#from networkx.algorithms import community
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import community as community_louvain
import matplotlib.cm as cm
#from sklearn.decomposition import PCA, KernelPCA
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.utils import resample

# # Data Loading
data = pd.read_csv(r"D:/Third Year/Data Science 1/Datasets/data.csv")
labels = pd.read_csv(r"D:/Third Year/Data Science 1/Datasets/labels.csv")
# # Merging data with labels
data['Class'] = labels['Class']

# Splitting data into features and target
X = data.drop(columns=['Class'])
y = data['Class']

###### Data Cleaning
# Preliminary Analysis
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Removing any rows with missing values for simplicity (you can also impute them)
data = data.dropna()

# Check for duplicates
data = data.drop_duplicates()

####### Distribution of Tumor Types
# Descriptive Statistics
print(data.describe())

# Class Distribution
print(data['Class'].value_counts())

# Visualizing the distribution of tumor types

# Convert 'Class' to a categorical type
data['Class'] = data['Class'].astype('category')

# Create the countplot with different colors
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
sns.countplot(x='Class', data=data, palette='Set2')  # 'Set2' is an example palette

# Add a title to the plot
plt.title('Distribution of Tumor Types')

# Show the plot
plt.show()

########### Distribution of Gene Expressions

selected_genes = ['gene_1', 'gene_2', 'gene_3','gene_4']  # replace with some actual gene names
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjusting for 2 rows and 2 columns
axes = axes.flatten()  # Flatten the axes array for easy indexing

for i, gene in enumerate(selected_genes):
    sns.kdeplot(data=data, x=gene, hue='Class', ax=axes[i])
    axes[i].set_title(f'Distribution of {gene} across tumor types')
    axes[i].grid(False)  # Removing grid lines

plt.tight_layout()
plt.show()

############### Correlation Analysis
# Selecting a random subset of genes for correlation analysis
np.random.seed(0)  # for reproducibility
sampled_genes = np.random.choice(X.columns, size=100, replace=False)  # selecting 100 genes randomly

# Calculating the correlation matrix for the sampled genes
sampled_correlation_matrix = data[sampled_genes].corr()

# Plotting the correlation matrix for the sampled genes
plt.figure(figsize=(12, 10))
sns.heatmap(sampled_correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Sampled Gene Expressions')
plt.xlabel('Sampled Genes')
plt.ylabel('Sampled Genes')
plt.show()

##### Checking if the dataset has some linearlity using  PCA vs Kernel PCA
# Standardize the data
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

# Assume X_scaled is your scaled feature matrix and y is the categorical class labels from your data
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scale)

# Apply Kernel PCA
kernel_pca = KernelPCA(n_components=2, kernel='rbf') # can try different kernels like 'rbf', 'poly', 'sigmoid'
X_kernel_pca = kernel_pca.fit_transform(X_scale)

# Plot the results of PCA and Kernel PCA with colors based on the categories
plt.figure(figsize=(16, 8))

# PCA Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, legend='full')
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Kernel PCA Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_kernel_pca[:, 0], y=X_kernel_pca[:, 1], hue=y, legend='full')
plt.title('Kernel PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show legend and plot
plt.legend()
plt.tight_layout()
plt.show()

######## Downsampling the Data to balance the clases 

# Separate the dataset into a dictionary with keys as class labels and values as the subset dataframes
class_groups = {cls: X[y == cls] for cls in y.unique()}

# Downsample each class to 78 samples
downsampled_groups = [resample(class_groups[cls], 
                               replace=False,    # sample without replacement
                               n_samples=78,     # to match minority class
                               random_state=42)  # for reproducible results
                     for cls in class_groups]

# Combine the downsampled dataframes
downsampled_data = pd.concat(downsampled_groups)

# Get the new class labels for the downsampled data
downsampled_labels = y.loc[downsampled_data.index]

# Check the new class distribution
print(downsampled_labels.value_counts())

# Add the labels back to the features
downsampled_data['Class'] = downsampled_labels

# To get the downsampled data without the labels
downsampled_data_without_labels = downsampled_data.drop(columns=['Class'])

## Model Evaluation Function

def evaluate_model(model, X, y, is_onehot):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if is_onehot:  # If the target is one-hot encoded, convert predictions for accuracy scoring
        y_test = label_encoder.inverse_transform([np.argmax(y) for y in y_test])
        predictions = label_encoder.inverse_transform([np.argmax(y) for y in predictions])

    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    accuracy = accuracy_score(y_test, predictions)
    test_error = 1 - accuracy

    return precision, recall, f1, accuracy, test_error

###### Tree-based feature Selection (Random Forest)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(downsampled_data_without_labels)

# One-hot encode the target variable
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(downsampled_labels.values.reshape(-1, 1))

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(downsampled_labels)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_onehot)

importances = rf.feature_importances_

# Selecting features above a certain threshold
threshold = 0.003  # Example threshold
selected_features = downsampled_data_without_labels.columns[importances > threshold]

X_tree_selected = downsampled_data_without_labels[selected_features]
print(X_tree_selected)

######## Feature importance from Random Forest

# Filter the feature importances for the selected features only
selected_importances = importances[X.columns.isin(X_tree_selected.columns)]

# Convert the feature importances to a pandas DataFrame
feature_importances_df = pd.DataFrame({
    'Feature': X_tree_selected.columns,
    'Importance': selected_importances
})

# Sort the DataFrame by importance score in descending order
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting the feature importances with a different color palette
plt.figure(figsize=(6, 4))
ax=sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importances_df.head(20),  # Show top 20 features
    palette=sns.color_palette("hsv", len(feature_importances_df.head(20)))  # Use a hue-saturation-value color palette
)
ax.grid(False)  # This will remove the gridlines
plt.title('Top 30 Features importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature Names')
plt.tight_layout()  # Fit the plot within the figure neatly
plt.show()

## Cluster heatmap for top 20 

# Assuming 'importances' contains importances for all features and 'X' is the original features DataFrame

# Sort by importance and select the top 20 features
top_features_df = feature_importances_df.sort_values('Importance', ascending=False).head(50)

# Get the names of the top 20 features
top_features = top_features_df['Feature'].values

# Subset your data to include only the top 20 features
X_top_features = X[top_features]

# Perform clustering
row_clusters = linkage(X_top_features.transpose(), method='ward', metric='euclidean')
col_clusters = linkage(X_top_features.T, method='ward', metric='euclidean')

# Create a clustermap
sns.clustermap(X_top_features.transpose(), row_linkage=row_clusters, col_linkage=col_clusters,
               figsize=(8, 8), cmap='viridis')

# Show the plot
plt.show()

# # Model Evaluation for Selected Features (Tree based)

# Tree-based models with one-hot encoded target
tree_based_models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

# Other models with label-encoded target
other_models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}


# Initialize a DataFrame to store all metrics
all_metrics_df = pd.DataFrame(columns=["Model", "Precision", "Recall", "F1_Score", "Accuracy", "Test_Error"])

# List to store each model's metrics
all_metrics_list = []

# Evaluate tree-based models
for name, model in tree_based_models.items():
    precision, recall, f1, accuracy, test_error = evaluate_model(model, X_tree_selected, y_onehot, is_onehot=True)
    all_metrics_list.append({
        "Model": name, "Precision": precision, "Recall": recall, "F1_Score": f1, "Accuracy": accuracy, "Test_Error": test_error
    })

# Evaluate other models
X_tree_selected = np.array(X_tree_selected, order='C')
y_encoded = np.array(y_encoded, order='C')
for name, model in other_models.items():
    precision, recall, f1, accuracy, test_error = evaluate_model(model, X_tree_selected, y_encoded, is_onehot=False)
    all_metrics_list.append({
        "Model": name, "Precision": precision, "Recall": recall, "F1_Score": f1, "Accuracy": accuracy, "Test_Error": test_error
    })

# Convert list of dicts to DataFrame
all_metrics_df = pd.DataFrame(all_metrics_list)
         
# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Melting the DataFrame to plot using seaborn
all_metrics_long = pd.melt(all_metrics_df, id_vars=["Model"], var_name="Metric", value_name="Value")

# Format the values in the DataFrame to six decimal places
all_metrics_df.update(all_metrics_df.select_dtypes(include=['float']).applymap('{:.6f}'.format))

# Create the figure and axes
fig = plt.figure(figsize=(12, 10))  # Adjust the size as needed to give more space

# Add a subplot for the bar plot with adjusted height for the plot
ax_bar = fig.add_subplot(211)
ax_bar.set_title('Model Performance ', fontsize=16)  # Add a title to the bar plot

# Create the bar plot on the first subplot with adjusted x-tick frequency
sns.barplot(x="Value", y="Model", hue="Metric", data=all_metrics_long, ax=ax_bar)
ax_bar.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend out of the plot
ax_bar.set_xlim(0, 1)  # Adjust x-axis limit if needed
ax_bar.set_xticks(np.arange(0, 1.1, 0.2))  # Change the x-tick frequency to 0.4
ax_bar.set_xlabel('')  # Remove x-axis label
ax_bar.set_ylabel('')  # Remove y-axis label


# Add a subplot for the table
ax_table = fig.add_subplot(212)
ax_table.axis('off')  # Hide spines and ticks

# Create the table and add it to the subplot
cell_text = all_metrics_df.values.tolist()
table = ax_table.table(
    cellText=cell_text,
    colLabels=all_metrics_df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]  # Use bbox to fit the table within the subplot
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # You may need to adjust the scale to fit the text

# Bold the headers and model names
for (row, col), cell in table.get_celld().items():
    if row == 0 or col == 0:  # Header row or first column (Model names)
        cell.get_text().set_fontweight('bold')
        cell.set_facecolor('lightgrey')
        
# Adjust layout to make room for both plot and table
plt.tight_layout()

# Save the figure
plt.savefig('RF_combined_plot_and_table.png', bbox_inches='tight')  # Use bbox_inches='tight' to include the whole figure

# Show the figure
plt.show()

# # Column and Row-wise clusterings in heatmap¶

# Calculate the linkage for rows and columns
row_clusters = linkage(X_tree_selected, method='ward', metric='euclidean')
col_clusters = linkage(X_tree_selected.T, method='ward', metric='euclidean')

# Create a clustermap with seaborn
sns.clustermap(X_tree_selected, 
               row_linkage=row_clusters, 
               col_linkage=col_clusters, 
               figsize=(10, 10),  # Adjust this as needed
               cmap='viridis')

# Display the heatmap
plt.show()

# # Genes Interation
X_tree_select = downsampled_data_without_labels[selected_features]
print(X_tree_select)

# Assuming 'X_tree_select' is your DataFrame
correlation_matrix = X_tree_select.corr()

# Set a threshold for correlation
threshold = 0.5

# Create a network graph
G = nx.Graph()

# Add nodes to the graph with original feature names as labels
for column in X_tree_select.columns:
    G.add_node(column, label=column)  # Use the 'label' attribute to store the original feature name

# Add edges to the graph if absolute correlation is greater than the threshold
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j])

# Specify the filename for saving the GraphML file
graphml_filename = "D:/Third Year/Data Science 1/network_graph.graphml"

# Save the graph as a GraphML file
nx.write_graphml(G, graphml_filename)

print(f"Graph saved as {graphml_filename}")

# Detect communities
partition = community_louvain.best_partition(G)
communities = set(partition.values())

# Create a color map for communities
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

# Calculate node sizes based on degree of the nodes
degrees = dict(G.degree)
node_sizes = [v * 50 for v in degrees.values()]  # Multiply by a factor to scale the sizes appropriately

node_sizes_dict = {node: size for node, size in zip(G.nodes(), node_sizes)}

# Draw the network with community color coding and node size variation
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
for i in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == i]
    labels = {node: G.nodes[node]['label'] for node in list_nodes}  # Get labels from the 'label' attribute
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size=[node_sizes_dict[node] for node in list_nodes],
                           node_color=cmap(i))
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title('Gene Interaction Network with Community Detection')
plt.axis('off')  # Turn off the axis
plt.show()

##Extracting genes in each community

# Create a dictionary to store genes in each community
genes_in_communities = {community_id: [] for community_id in set(partition.values())}

# Iterate through nodes and add genes to their respective communities
for node, community_id in partition.items():
    genes_in_communities[community_id].append(G.nodes[node]['label'])

# Print genes in each community
for community_id, genes in genes_in_communities.items():
    print(f"Community {community_id}: {', '.join(genes)}")


###########################
#Test for the Communities

# Step 2: Compute the Original Community Structure
correlation_matrix = pd.DataFrame(X_tree_selected).corr()
threshold = 0.5
original_G = nx.Graph()

for i in range(correlation_matrix.shape[0]):
    for j in range(i+1, correlation_matrix.shape[1]):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            original_G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j])

original_partition = community_louvain.best_partition(original_G)
original_modularity = community_louvain.modularity(original_partition, original_G)

# Step 3: Permutation Test
num_permutations = 1000
modularity_scores = []

for _ in range(num_permutations):
    # Randomize the network while preserving degree distribution
    randomized_G = nx.expected_degree_graph([original_G.degree(n) for n in original_G.nodes()], selfloops=False)
    
    # Recompute the community structure on the randomized network
    randomized_partition = community_louvain.best_partition(randomized_G)
    randomized_modularity = community_louvain.modularity(randomized_partition, randomized_G)
    
    modularity_scores.append(randomized_modularity)


# Ensure modularity_scores is a list of numeric values and not a list of lists
modularity_scores = [score for score in modularity_scores if isinstance(score, (int, float))]
p_value = np.sum(np.array(modularity_scores) >= original_modularity) / num_permutations


# Output the results
print(f"Original Modularity: {original_modularity}")
print(f"Average Randomized Modularity: {np.mean(modularity_scores)}")
print(f"p-value: {p_value}")

# If you want to visualize the distribution of randomized modularity scores
plt.hist(modularity_scores, bins=30, color='blue', edgecolor='black')
plt.axvline(original_modularity, color='red', linestyle='dashed', linewidth=2)
plt.title('Distribution of Modularity Scores from Randomized Networks')
plt.xlabel('Modularity Score')
plt.ylabel('Frequency')
plt.show()

# Interpret the results
if p_value < 0.05:
    print("The community structure is significantly different from random networks.")
else:
    print("The community structure is not significantly different from random networks.")

if original_modularity > 0.3:
    print("The original network has a strong community structure.")
else:
    print("The original network does not have a strong community structure.")


