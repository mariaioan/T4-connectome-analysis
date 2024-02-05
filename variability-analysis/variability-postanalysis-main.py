# -*- coding: utf-8 -*-
"""
Created on Friday 25 16:40:16 2023

@author: smolina

variability postanalysis of presynaptic inputs
"""
#%% Importing packages
import os
import pandas as pd
import glob
import pandas as pd
import numpy as np
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from scipy.stats import kruskal, bartlett
from scikit_posthocs import posthoc_dunn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.metrics import pairwise_distances
import matplotlib.gridspec as gridspec
from sklearn.cluster import AgglomerativeClustering
from fafbseg import flywire
import navis


#%% Custom functions 

#Importing custom functions from helper file
from helper import cosine_similarity_and_clustering, remove_outliers, perform_levene_test, determine_subgroup


#%% 
############################################# USER INFORMATION ################################################
###############################################################################################################

# Specify the folder containing files (processed-data)
PC_disc = 'D'
dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Processed-data' # Path to the PROCESSED_DATA folder
dataPath_raw =  f'{PC_disc}:\Connectomics-Data\FlyWire\Excels\drive-data-sets\submission_nature' # Path to the RAW_DATA folder
fig_save_path = os.path.join(dataPath,"Figures")
save_figures = True
exclude_outliers = False


# Comparisons (between processed-data)
single_data_set = True
data_frames_to_compare_ls = ['Tm9_FAFB_L_R_'] # ['Tm9_300_healthy_L3_L_R_20230823'], ['Tm9_FAFB_R_'], ['Tm9_FAFB_L_R_'], ['Tm9_FAFB_R_', 'Tm1_FAFB_R_','Tm2_FAFB_R_']
user_defined_categoriers = ['Tm9_L_R'] # ['Tm9_R'] , ['Tm9_L_R'],  ['Tm9_R', 'Tm1_R', 'Tm2_R']
dataset_subgroups = ['R', 'L']# ['D', 'V'], ['R', 'L']
dataset = ['FAFB_R_L'] # ['FAFB_R_L'], ['FAFB_R']
subgroups_name = 'hemisphere' # 'dorso-ventral', hemisphere
_sheet = '_Relative_counts' #  '_Absolut_counts', '_Relative_counts'

# K-mean clustering
n_k_clusters = 2 # Number of desired clusters

# Columns location
neuron_of_interest = 'Tm9'
fileDate = '20230823'
fileName_database = f'{neuron_of_interest} proofreadings_{fileDate}.xlsx'
filePath = os.path.join(dataPath_raw,fileName_database)
database_df = pd.read_excel(filePath)


excel_file_to_load = []
for file_name in data_frames_to_compare_ls:
    file_name = file_name + '.xlsx'
    excel_file_to_load.append(os.path.join(dataPath,file_name))


#%% 
######################################## LOADING PREPROCESSED DATA ############################################
###############################################################################################################

# Get a list of all Excel files in the folder
excel_files = glob.glob(os.path.join(dataPath , '*.xlsx'))

# Initialize an empty dictionary to store DataFrames
data_frames = {}

# Iterate through each Excel file
for excel_file in excel_file_to_load:
    # Get the distinct part of the filename (excluding extension)
    file_name = os.path.splitext(os.path.basename(excel_file))[0]
    
    # Load all sheets from the Excel file into a dictionary of DataFrames
    sheet_dataframes = pd.read_excel(excel_file, sheet_name=None,index_col = 0)
    
    # Iterate through each sheet DataFrame
    for sheet_name, sheet_df in sheet_dataframes.items():
        # Create a key for the combined name of DataFrame
        df_name = f"{file_name}_{sheet_name}"
        
        # Store the DataFrame in the dictionary
        data_frames[df_name] = sheet_df

# # Now you can access the DataFrames by their combined names
# for df_name, df in data_frames.items():
#     print(f"DataFrame Name: {df_name}")
#     print(df)  # Print the DataFrame
#     print("\n")


#%% 
############################################### DATA ANALYSIS #################################################
###############################################################################################################


############################################# Synapse count variation #########################################

## Synapse count distributions for the chosen _sheet

# Initialize an empty DataFrame
syn_count_df = pd.DataFrame()

# For single data frames
if single_data_set:
    subgroups_data_frames = {}

    # Loop through each letter and create a DataFrame for that letter
    for letter in dataset_subgroups:
        subgroups_data_frames[letter] = sheet_df[sheet_df.index.str.contains(letter)]
    
    # Find the maximum length among all lists
    max_length = max(len(_data) for _data in subgroups_data_frames.values())

    # Iterate over each DataFrame
    for i, df_name in enumerate(dataset_subgroups):
        _data = subgroups_data_frames[df_name]
        
        # Sum all columns along the rows to get 'total_count'
        _total_count = _data.sum(axis=1).tolist()
        
        # Add a new column with NaN values if the length is less than the maximum length
        syn_count_df[dataset_subgroups[i]] = _total_count + [np.nan] * (max_length - len(_total_count))

# For many data frames
else: 

    # Find the maximum length among all lists
    max_length = max(len(_data) for _data in data_frames.values())

    # Iterate over each DataFrame
    for i, df_name in enumerate(data_frames_to_compare_ls):
        df_name = df_name + _sheet
        _data = data_frames[df_name]
        
        # Sum all columns along the rows to get 'total_count'
        _total_count = _data.sum(axis=1).tolist()
        
        # Add a new column with NaN values if the length is less than the maximum length
        syn_count_df[user_defined_categoriers[i]] = _total_count + [np.nan] * (max_length - len(_total_count))



# ##################################    Cosine similarity in absolute counts    #################################

## For multiple data sets
# Computing cosine similarity for absolute counts
combined_cosine_sim_summary_df = pd.DataFrame()
cos_sim_medians_dict = {}
for i,df_name in enumerate(data_frames_to_compare_ls):
    df_name = df_name + _sheet
    _data = data_frames[df_name]
    # Call the function and only define cosine_sim_summary_df
    cosine_sim_summary_df = cosine_similarity_and_clustering(_data,dataset_subgroups)[1]
    cosine_sim_summary_df['neuron'] = user_defined_categoriers[i]
    #Call the function and only define cos_sim_medians 
    cos_sim_medians_dict[df_name] = cosine_similarity_and_clustering(_data,dataset_subgroups)[8]

    # Concatenate the current dataframe to the combined dataframe
    combined_cosine_sim_summary_df = pd.concat([combined_cosine_sim_summary_df, cosine_sim_summary_df])

# Reset index of the combined dataframe
combined_cosine_sim_summary_df = combined_cosine_sim_summary_df.reset_index(drop=True)

## For single data set
if single_data_set:
    _dict = cos_sim_medians_dict[data_frames_to_compare_ls[0]+_sheet]

    combined_cosine_sim_list = []
    combined_neuron_list = []
    for key,value in _dict.items():
        combined_cosine_sim_list = combined_cosine_sim_list + value
        combined_neuron_list = combined_neuron_list + [key]*len(value)

    # Create the dataframe with all subgropus
    combined_cosine_sim_summary_df = pd.DataFrame()
    combined_cosine_sim_summary_df['cosine_sim'] = combined_cosine_sim_list
    combined_cosine_sim_summary_df['neuron'] = combined_neuron_list


########################################    Counts in single data frame   #####################################


## For single data set
if single_data_set:
    sheet_df = data_frames[data_frames_to_compare_ls[0]+_sheet]
    sheet_df .fillna(0, inplace=True)
    data_df = sheet_df # Seb  coding here
    
    # Apply the function to create the "dorso-ventral" column
    sheet_df [subgroups_name] = sheet_df .index.map(lambda x: determine_subgroup(x, dataset_subgroups))


#################################  Cluster analysis in single data frame   ####################################
###############################################################################################################

###########################################  K-mean clustering   ##############################################

## For single data set
if single_data_set:
    print('Cluster analysis in single data frame')

    data = data_df.to_numpy()
    order = ['L3','Mi4','CT1','Tm16','Dm12','Tm20',
            'C3','Tm1','PS125','L4','ML1','TmY17','C2',
            'OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
    data_df = data_df[order].copy()
    number_of_columns = len(data_df)


    # Perform K means with the desired cluster n
    kmeans = KMeans(
        init="random",
        n_clusters=n_k_clusters,
        n_init=10,
        max_iter=300,
        random_state=42)
    kmeans.fit(data_df)
    unique_clusters, counts = np.unique(kmeans.labels_, return_counts=True)
    print(f'Neurons per cluster: {counts}')

    # Add a new column to the original dataset with the cluster labels
    data_df['Cluster'] = kmeans.labels_

    # Explore cluster characteristics
    cluster_means = data_df.groupby('Cluster').mean()

    # Visualize the clusters (for 2D data)
    feature_1 = 0
    feature_2 = 1


    # Get the data points for the selected features
    selected_data = data_df.iloc[:, [feature_1, feature_2]].values

    # Compute the silhouette score
    silhouette_avg = silhouette_score(selected_data, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}")

###########################################  Hamming clustering   ##############################################
neurons_to_include = ['Tm16','Dm12','Tm20',
            'C3','Tm1','PS125','L4','ML1','TmY17','C2',
            'OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
cluster_df = data_df[neurons_to_include].copy()
data_cluster = cluster_df.to_numpy()

## Hamming distances
# Calculate pairwise Hamming distances
binary_array = (data_cluster>0).astype(int)

distances = pairwise_distances(binary_array, metric="hamming")

# Perform hierarchical clustering using linkage
linkage_matrix = hierarchy.linkage(distances, method='ward')  # You can choose different linkage methods


# Reorder data based on dendrogram leaves
reordered_indices = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
reordered_array = data_cluster[reordered_indices]
reordered_distances = pairwise_distances(reordered_array, metric="hamming")

# Clustering 
selected_cluster_n = 7
clusterer = AgglomerativeClustering(n_clusters=selected_cluster_n, linkage='ward')
cluster_labels = clusterer.fit_predict(distances)

unique_clusters , counts = np.unique(cluster_labels, return_counts= True)



#########################################  PCA analysis  ######################################################
###############################################################################################################
# NaN means the Tm9 neuron did not receive any input from that neuron
neurons_to_include = ['L3','Mi4','CT1','Tm16','Dm12','Tm20',
            'C3','Tm1','PS125','L4','ML1','TmY17','C2',
            'OA-AL2b2','Tm2','Mi13','putative-fru-N.I.','Tm5c','Me-Lo-2-N.I.','TmY15']
pca_df = data_df[neurons_to_include].copy()

pca_data= pca_df.fillna(0) # replace NaN with 0s
pca_data_array = pca_data.to_numpy(dtype=float,copy=True)

# Standardize
# Features are in rows now (due to transpose)
pca_data_array_norm = pca_data_array-pca_data_array.mean(axis=0)
pca_data_array_norm /= pca_data_array_norm.std(axis=0)
n = pca_data_array_norm.shape[0]

# Cov matrix and eigenvectors
cov = (1/n) * pca_data_array_norm.T @ pca_data_array_norm
eigvals, eigvecs = np.linalg.eig(cov)
k = np.argsort(eigvals)[::-1]
eigvals = eigvals[k]
eigvecs = eigvecs[:,k]



#%% 
############################################ PLOTS and STATISTICS #############################################
###############################################################################################################


########################################### Synapse count variability ########################################
##############################    Leven test for equality of variances    ####################################



# Plotting
# Plot box plots and histograms in two subplots
_binwidth = 6
# Removing outliers
if exclude_outliers:
    syn_count_df = remove_outliers(syn_count_df, multiplier=1.5)

# Calculate the coefficient of variation (CV) for each column
cv_values = syn_count_df.std() / syn_count_df.mean()

# Perform F-test for equality of variances
f_test_results = f_oneway(*[syn_count_df[col].dropna() for col in syn_count_df.columns])

# Perform Bartlett's test for equality of variances
bartlett_test_results = bartlett(*[syn_count_df[col].dropna() for col in syn_count_df.columns])

# Perform Levene's test for equality of variances pairwise with Bonferroni correction
column_combinations = list(combinations(syn_count_df.columns, 2))
alpha = 0.05  # Set your desired significance level

# Create subplots for box plots and histograms
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plots with the same colors used in histograms
sns.boxplot(data=syn_count_df, ax=axes[0], palette=sns.color_palette('husl', n_colors=len(syn_count_df.columns)))
axes[0].set_title("Synapse count variability (Levene's Test)")
axes[0].set_ylabel('Synapse counts')

# Add CV values to the box plots
for i, col in enumerate(syn_count_df.columns):
    axes[0].text(i, syn_count_df[col].max() + 10, f'CV={cv_values[col]:.2f}', ha='center', va='bottom', color='blue')

# Plot horizontal lines with p-values
for i, (col1, col2) in enumerate(column_combinations):
    p_value = perform_levene_test(syn_count_df[col1], syn_count_df[col2],column_combinations)

    print(f"Levene's Test for {col1} and {col2} p-value (Bonferroni corrected): {p_value:.4f}")
    print("Significant" if p_value < alpha else "Not significant")

    # Extract x-axis tick locations for each column
    ticks = axes[0].get_xticks()
    
    # Find the index of the current columns in the list of ticks
    index_col1 = syn_count_df.columns.get_loc(col1)
    index_col2 = syn_count_df.columns.get_loc(col2)
    
    # Calculate the center positions based on the tick locations
    center1 = ticks[index_col1]
    center2 = ticks[index_col2] 
    
    y_position = max(syn_count_df[col1].max(), syn_count_df[col2].max()) + 20
    
    # Plot horizontal lines from one boxplot center to the other
    axes[0].hlines(y=y_position, xmin=center1, xmax=center2, color='red', linewidth=2)
    axes[0].text((center1 + center2) / 2, y_position + 2, f'p={p_value:.4f}', ha='center', va='bottom', color='red')

# Histograms for each column without outliers using Seaborn with the same colors
for col_idx, (col, color) in enumerate(zip(syn_count_df.columns, sns.color_palette('husl', n_colors=len(syn_count_df.columns)))):
    sns.histplot(data=syn_count_df[col], binwidth=_binwidth, alpha=0.5, ax=axes[1], kde=True, label=col, color=color)

axes[1].set_title('Synapse count variability')
axes[1].set_xlabel('Synapse counts')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Save the figure if required
if save_figures:
    figure_title = f'\Synaptic_count_variability_no_ouliers_{user_defined_categoriers}{_sheet}.pdf'
    plt.savefig(fig_save_path + figure_title)
    plt.close()




############################################    Cosine similarity     #########################################
###########################################    Multiple Comparisons    ########################################

data = combined_cosine_sim_summary_df.copy()

### Chekcing data distribution
# Check if the data in each category is normally distributed using the Shapiro-Wilk test:
categories = data["neuron"].unique()
normality_results = {}

for category in categories:
    category_data = data[data["neuron"] == category]["cosine_sim"]
    _, p_value = shapiro(category_data)
    normality_results[category] = p_value

print("Shapiro-Wilk p-values for normality:")
print(normality_results)

### Perform One-Way ANOVA and Multiple Comparisons:
# Perform one-way ANOVA and then use the Tukey HSD test for multiple comparisons if the data is normally distributed:
anova_results = f_oneway(*[data[data["neuron"] == category]["cosine_sim"] for category in categories])

if all(p > 0.05 for p in normality_results.values()):
    print("One-Way ANOVA p-value:", anova_results.pvalue)
    tukey_results = pairwise_tukeyhsd(data["cosine_sim"], data["neuron"])
    print(tukey_results)
else:
    print("Data is not normally distributed. Performing Kruskal-Wallis test.")
    kruskal_results = kruskal(*[data[data["neuron"] == category]["cosine_sim"] for category in categories])
    print("Kruskal-Wallis p-value:", kruskal_results.pvalue)

    # Applying Dunn-Bonferroni correction for multiple comparisons
    dunn_results = posthoc_dunn(data, val_col="cosine_sim", group_col="neuron", p_adjust="bonferroni")
    print(dunn_results)

# Plot box plots with p-values
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="neuron", y="cosine_sim")
plt.title("Cosine Similarity by Neuron Category")
plt.ylabel("Cosine Similarity")
plt.xlabel("Neuron Category")

# Adding p-values to the plot
comparison_results = tukey_results if "tukey_results" in locals() else dunn_results

# Adding lines and p-values to the plot
line_distance = 0.05  # Adjust this value to increase the distance between lines

line_positions = {}  # Store line positions for each comparison

for i, category1 in enumerate(categories):
    for j, category2 in enumerate(categories):
        if j > i:  # Avoid redundant comparisons
            y_pos1 = max(data[data["neuron"] == category1]["cosine_sim"]) + 0.02
            y_pos2 = max(data[data["neuron"] == category2]["cosine_sim"]) + 0.02
            y_line = max(y_pos1, y_pos2) + (line_distance * len(line_positions))
            line_positions[(i, j)] = y_line
            
            # Calculate x position for the line
            x_pos = (i + j) / 2
            
            # Access p-values based on the analysis performed
            if "tukey_results" in locals():
                p_value = tukey_results.pvalues[i, j]
            else:
                p_value = comparison_results.loc[category1, category2]
            
            # Draw line and add p-value text
            plt.plot([i, j], [y_line, y_line], linewidth=1, color='black')
            plt.text(x_pos, y_line, f"p = {p_value:.4f}", ha='center')

# Adjust ylim to fit the lines and p-values
plt.ylim(plt.ylim()[0], max(line_positions.values()) + line_distance)

if save_figures:
    plt.savefig(f'{fig_save_path}\Cosine_similarity_{user_defined_categoriers}{_sheet}.pdf')
    plt.close()

########################################    Counts for single data frames    #####################################
##Plotting:
if single_data_set:

    # Filter out the "dorso-ventral" column
    data_cols = sheet_df .columns[:-1]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize a list to store violin plot data
    violin_plot_data = []

    # Create an offset for the dodge effect
    offset = 0

    # Iterate through each data column
    for i, col in enumerate(data_cols):
        # Create a DataFrame for the current data column and "dorso-ventral" column
        data = sheet_df [[col, subgroups_name]]
        
        # Create a violin plot for the current data column
        sns.violinplot(x=subgroups_name, y=col, data=data, ax=ax, position=i+offset)
        
        # Update the offset for the next plot
        offset += 0.2  # Adjust the value as needed
        
        # Append the data to the list
        violin_plot_data.append(data)

    # Set the x-axis label
    ax.set_xlabel(subgroups_name)
    ax.set_ylabel('Synaptic count (%)')

    # Set the title
    ax.set_title("Violin Plots for Each Column")


    ##Statistics. Pair-wise comparison between subgroups in each cell type
    # !!! So far meant for just 2 subgroups only

    data_cols = sheet_df .columns[:-1]

    # Create an empty list to store p-values
    p_values_list = []

    # Iterate through each data column
    for col in data_cols:
        # Get unique dorso-ventral categories
        categories = sheet_df [subgroups_name].unique()
        
        # Initialize a list to store p-values for the current data column
        p_values_col = []
        
        # If only one category exists, append None to p_values_col and continue
        if len(categories) == 1:
            p_values_col.append(None)
            p_values_list.append(p_values_col)
            continue
        
        # Generate combinations of categories for pairwise comparison
        category_combinations = combinations(categories, 2)
        
        # Iterate through category combinations
        for cat1, cat2 in category_combinations:
            group1 = sheet_df [sheet_df [subgroups_name] == cat1][col]
            group2 = sheet_df [sheet_df [subgroups_name] == cat2][col]
            
            # Perform the Shapiro-Wilk test for normality
            _, p_value1 = stats.shapiro(group1)
            _, p_value2 = stats.shapiro(group2)
            
            # Decide whether to use parametric or non-parametric test based on normality
            if p_value1 > 0.05 and p_value2 > 0.05:
                t_statistic, p_value = stats.ttest_ind(group1, group2)
                print(f'{col} is normally distributed')
            else:
                _, p_value = stats.mannwhitneyu(group1, group2)
            
            p_values_col.append(p_value)
        
        p_values_list = p_values_list + p_values_col

    # Convert the p-values list to a DataFrame
    p_values_df = pd.DataFrame()
    p_values_df['Neuron'] = data_cols.tolist()
    p_values_df['p_value'] = p_values_list

    # Display the p-values DataFrame
    print(f'Significant difference between {subgroups_name}')
    print(p_values_df[p_values_df["p_value"]<0.05])

    if save_figures:
        figure_title = f'\Testing_violin_plots{_sheet}.pdf'
        fig.savefig(fig_save_path+figure_title)

################################################## Plotting clusters ######################################

if single_data_set:
    print('Plotting box plots for each cluster')
    ## Plotting
    # Calculate the number of rows and columns for subplots
    num_rows = len(unique_clusters)
    num_cols = 1  # Assuming a single column for simplicity, adjust as needed

    # Calculate the figure size based on the number of rows
    fig_height = 2 * num_rows  # Adjust the multiplier as needed
    fig_size = (8, fig_height)

    # Adjust subplot spacing and rotate x-axis labels
    fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, sharey=True, gridspec_kw={'hspace': 0.5})
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Neurons per cluster (n = {number_of_columns}): {counts} \n {data_frames_to_compare_ls}')

    #major_inputs_data = data_df[['L3', 'Mi4', 'CT1', 'Tm16', 'Dm12']]
    major_inputs_data = data_df[order]

    for ax, cluster in zip(axs, unique_clusters):
        sns.boxplot(data=major_inputs_data.iloc[np.where(kmeans.labels_ == cluster)[0]], ax=ax)
        ax.set_ylabel('Relative Counts')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate x-axis labels
        
   
    if save_figures:
        figure_title = f'\Clustered_data_{data_frames_to_compare_ls}{_sheet}_{n_k_clusters}clusters.pdf'
        fig.savefig(fig_save_path+figure_title)
        plt.close()

######################################## Plotting Hamming distance heatmap #################################

if single_data_set:
    # Create a figure with custom grid layout
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[8, 1], height_ratios=[1.2, 8, 0.5])

    # Plot the dendrogram_cosine
    ax_dendrogram_hamming = plt.subplot(gs[0, :-1])
    ax_dendrogram_hamming.spines['top'].set_visible(False)
    ax_dendrogram_hamming.spines['right'].set_visible(False)
    ax_dendrogram_hamming.spines['bottom'].set_visible(False)
    ax_dendrogram_hamming.spines['left'].set_visible(False)
    ax_dendrogram_hamming.get_xaxis().set_visible(False)
    ax_dendrogram_hamming.get_yaxis().set_visible(False)
    hierarchy.dendrogram(linkage_matrix, ax=ax_dendrogram_hamming, color_threshold=0)

    # Plot the heatmap using the reordered DataFrame
    ax_heatmap = plt.subplot(gs[1, :-1])
    sns.heatmap(reordered_distances, cmap='rocket_r', annot=False, xticklabels=cluster_df.index, yticklabels=cluster_df.index, ax=ax_heatmap, cbar=False)

    ax_heatmap.set_xlabel('Column')
    ax_heatmap.set_ylabel('Column')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, fontsize=3)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=3)

    # Create a dummy plot for the color bar
    dummy_cax = fig.add_subplot(gs[2, :-1])
    dummy_cax.set_xticks([])
    dummy_cax.set_yticks([])

    # Add color bar below the heatmap
    cbar = plt.colorbar(ax_heatmap.collections[0], cax=dummy_cax, orientation='horizontal')
    cbar.set_label('Hamming distance')

    # Clusters plot
    plot_df = cluster_df
    binarized_df = plot_df.applymap(lambda x: 1 if x > 0 else 0)
    fig2, axs2 = plt.subplots(selected_cluster_n,figsize=(6, 12))
    for icluster, cluster_n in enumerate(unique_clusters):
        current_c_df = binarized_df.iloc[cluster_labels==cluster_n]
        sns.heatmap(current_c_df,ax=axs2[icluster],cbar=False)
        
        axs2[icluster].set_title(f"Cluster{cluster_n}, n:{len(current_c_df)}")
        axs2[icluster].set_yticklabels([])
        if not((icluster == len(unique_clusters)-1)):
            axs2[icluster].set_xticklabels([])

    if save_figures:
        figure_title = f'\Hamming_distance_{data_frames_to_compare_ls}{_sheet}.pdf'
        fig.savefig(fig_save_path+figure_title)
        figure_title = f'\Hamming_distance_{data_frames_to_compare_ls}{_sheet}_{selected_cluster_n}clusters.pdf'
        fig2.savefig(fig_save_path+figure_title)
        plt.close()


################################################ Plotting PCA #################################################

if single_data_set:
    print('Plotting PCA analysis')
    # Explained variance of PCs
    #plot the square-root eigenvalue spectrum
    fig = plt.figure(figsize=[5,5])
    explained_var = np.cumsum(eigvals)/max(np.cumsum(eigvals))*100
    explained_var = np.roll(explained_var,1)
    explained_var[0] = 0
    explained_var= np.append(explained_var,(np.cumsum(eigvals)/max(np.cumsum(eigvals))*100)[-1])
    plt.plot(explained_var,'-o',color='black')
    plt.xlabel('dimensions')
    plt.ylabel('explained var (percentage)')
    plt.xlim([0,20])
    plt.ylim([0,100])
    plt.xticks(range(21))
    plt.title(f'Explained variances {np.around(explained_var[0:3],2)}...')
    fig.savefig(os.path.join(fig_save_path,f'PCA_varExplained{_sheet}.pdf'))
    plt.close()

    #PCA_data
    fig = plt.figure()
    pc_1 = 0
    pc_2 = 1
    plt.scatter(pca_data_array_norm @ eigvecs[:,pc_1],pca_data_array_norm @ eigvecs[:,pc_2])
    plt.xlabel(f'PC {pc_1+1}')
    plt.ylabel(f'PC {pc_2+1}')
    fig.savefig(os.path.join(fig_save_path,f'PCA_data{_sheet}.pdf'))
    plt.close()

    # %% Contributions
    fig = plt.figure(figsize=[4,6])
    plt.imshow(np.array([eigvecs[:,0],eigvecs[:,1]]).T,cmap='coolwarm',aspect='auto')
    # plt.imshow(np.array([eigvecs[:4,0],eigvecs[:4,1],eigvecs[:4,2],eigvecs[:4,3]]).T,cmap='coolwarm',aspect='auto')
    plt.colorbar()
    plt.xlabel('Principal components (PCs)')
    ax = plt.gca()
    a = list(range(0, eigvecs.shape[0]))
    ax.set_yticks(a)
    ax.set_yticklabels(pca_data.columns)
    plt.title('Contribution of neurons to PCs')
    fig.savefig(os.path.join(fig_save_path,f'PCA_PC_contributions{_sheet}.pdf'))
    plt.close()
    # %% Dorso ventral differences in PCA?
    delimiter = ":"

    # Extract letters after the delimiter for each string
    dv_labels = [string.split(delimiter)[3] if delimiter in string else "" for string in data_df.index]

    fig = plt.figure()
    pc_1 = 0
    pc_2 = 1
    pc1 = pca_data_array_norm @ eigvecs[:,pc_1]
    pc2 = pca_data_array_norm @ eigvecs[:,pc_2] 

    d_labels = ["D" in string for string in dv_labels]
    v_labels = ["V" in string for string in dv_labels]
    plt.scatter(pc1[d_labels],pc2[d_labels],color=[152/255,78/255,163/255,0.8],label='dorsal')
    plt.scatter(pc1[v_labels],pc2[v_labels],color=[77/255,175/255,74/255,0.8],label='ventral')
    plt.legend()
    plt.xlabel(f'PC {pc_1+1}')
    plt.ylabel(f'PC {pc_2+1}')
    fig.savefig(os.path.join(fig_save_path,f'PCA_DV{_sheet}.pdf'))
    plt.close()
    # %% R and Left
    delimiter = ":"

    # Extract letters after the delimiter for each string
    rl_labels = [string.split(delimiter)[2][0] if delimiter in string else "" for string in data_df.index]

    fig = plt.figure()
    pc_1 = 0
    pc_2 = 1
    pc1 = pca_data_array_norm @ eigvecs[:,pc_1]
    pc2 = pca_data_array_norm @ eigvecs[:,pc_2] 

    r_labels = ["R" in string for string in rl_labels]
    l_labels = ["L" in string for string in rl_labels]
    plt.scatter(pc1[r_labels],pc2[r_labels],color='r',label='right')
    plt.scatter(pc1[l_labels],pc2[l_labels],color='g',label='left')
    plt.legend()
    plt.xlabel(f'PC {pc_1+1}')
    plt.ylabel(f'PC {pc_2+1}')
    fig.savefig(os.path.join(fig_save_path,f'PCA_RL{_sheet}.pdf'))
    plt.close()
    # %% K means
    n_k_clusters = 2
    kmeans = KMeans(
        init="random",
        n_clusters=n_k_clusters,
        n_init=10,
        max_iter=300,
        random_state=42)
    kmeans.fit(pca_data_array)
    unique_clusters , counts = np.unique(kmeans.labels_, return_counts= True)
    print(f'Neurons per cluster: {counts}' )
    # %% Plot clujsters

    # Extract letters after the delimiter for each string

    fig = plt.figure()
    pc_1 = 0
    pc_2 = 1
    pc1 = pca_data_array_norm @ eigvecs[:,pc_1]
    pc2 = pca_data_array_norm @ eigvecs[:,pc_2] 

    plt.scatter(pc1[kmeans.labels_==0],pc2[kmeans.labels_==0],color=[228/255,26/255,28/255,0.8],label='Cluster 1')
    plt.scatter(pc1[kmeans.labels_==1],pc2[kmeans.labels_==1],color=[55/255,126/255,184/255,0.8],label='Cluster 2')
    plt.legend()
    plt.xlabel(f'PC {pc_1+1}')
    plt.ylabel(f'PC {pc_2+1}')
    fig.savefig(os.path.join(fig_save_path,f'PCA_KmeansClusters{_sheet}.pdf'))
    plt.close()


    # Cluster Visualization on the Medulla
    OL_labels = [string.split(":")[2] for string in data_df.index]
    ids= database_df["optic_lobe_id"]


    xyz = np.zeros([len(OL_labels),3])
    for idx, neuron in enumerate(OL_labels):
        df_loc = np.where(ids==neuron)[0]
        coordinate = database_df.iloc[df_loc]["XYZ-ME"].to_numpy(dtype=str, copy=True)
        xyz[idx,:] = np.array([coordinate[0].split(',')],dtype=float)
    xyz *=[4,4,40] # For plotting it using navis

    ##
    mesh_OL_L = 'ME_R' # This is correct for fafbseq version 1.14.0 and before 
    mesh_OL_R = 'ME_L' # This is correct for fafbseq version 1.14.0 and before 
    mesh_azim_L = -18# -18 for ME_R, 16 for ME_L
    mesh_elev_L = -148 # -148 for ME_R, -50 for ME_L
    mesh_azim_R = 16# -18 for ME_R, 16 for ME_L
    mesh_elev_R = -50 # -148 for ME_R, -50 for ME_L

    OL_R = flywire.get_neuropil_volumes([mesh_OL_R]) #['ME_R','LO_R','LOP_R']
    OL_L = flywire.get_neuropil_volumes([mesh_OL_L]) #['ME_R','LO_R','LOP_R']

    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    navis.plot2d([OL_R], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    ax.scatter(xyz[kmeans.labels_==0,0],xyz[kmeans.labels_==0,1],xyz[kmeans.labels_==0,2],'.',linewidth=0,s=15,color=[228/255,26/255,28/255,0.8],label='Cluster 1')
    ax.scatter(xyz[kmeans.labels_==1,0],xyz[kmeans.labels_==1,1],xyz[kmeans.labels_==1,2],'.',linewidth=0,s=15,color=[55/255,126/255,184/255,0.8],label='Cluster 2')
    ax.azim= mesh_azim_R
    ax.elev= mesh_elev_R
    fig.suptitle(f"Clusters on the R medulla")
    fig.savefig(os.path.join(fig_save_path,f'Location_KmeanClusters_R_{_sheet}.pdf'))
    plt.close()

    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    navis.plot2d([OL_L], method='3d_complex', ax=ax,view=(12, 2),scalebar = '10 um')
    ax.scatter(xyz[kmeans.labels_==0,0],xyz[kmeans.labels_==0,1],xyz[kmeans.labels_==0,2],'.',linewidth=0,s=15,color=[228/255,26/255,28/255,0.8],label='Cluster 1')
    ax.scatter(xyz[kmeans.labels_==1,0],xyz[kmeans.labels_==1,1],xyz[kmeans.labels_==1,2],'.',linewidth=0,s=15,color=[55/255,126/255,184/255,0.8],label='Cluster 2')
    ax.azim= mesh_azim_L
    ax.elev= mesh_elev_L
    fig.suptitle(f"Clusters on the L medulla")
    fig.savefig(os.path.join(fig_save_path,f'Location_KmeanClusters_L_{_sheet}.pdf'))
    plt.close()