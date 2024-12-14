# Install required libraries

!pip install wikipedia-api 

import os
import requests
from bs4 import BeautifulSoup
import wikipediaapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer

	
toys_df = pd.read_csv('advanced_toys_dataset.csv' )  
toys_df

print("Dataset Info:")
print("-" * 40)
print(toys_df.info())

print("\nSummary Statistics:")
print("-" * 40)
print(toys_df.describe())


# Drop rows where 'image_urls' is null
df_non_null = toys_df.dropna(subset=["image_urls"])

# Reset the index and drop the old index
toys_df = df_non_null.reset_index(drop=True)
toys_df
# Display the resulting DataFrdf_non_null
df = toys_df 

plt.subplot()
df['age_group_hint'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Age Group Distribution')
plt.show()

import pandas as pd


# Simulate loading your data
data =toys_df # Replace with your actual CSV path

# Ensure rows with NaN age_group_hint are dropped (clean data)
data.dropna(subset=["age_group_hint"], inplace=True)

# Define a heuristic rule function
def predict_age_group(topic_keywords):
    """
    Predict the age group based on heuristic rules.
    :param topic_keywords: Text related to the topic.
    :return: Predicted age group ('children', 'teen', 'adult')
    """
    # Convert text to lowercase for consistent keyword matching
    topic_keywords = topic_keywords.lower()
    
    # Rule-based heuristic logic
    if any(keyword in topic_keywords for keyword in ["toy", "playing", "child", "children"]):
        return "children"
    elif any(keyword in topic_keywords for keyword in ["game", "video game", "teen", "youth"]):
        return "teen"
    else:
        return "adult"


# Apply heuristic logic only to rows labeled 'unspecified'
data["predicted_age_group"] = data["topic_keywords"].apply(
    lambda x: predict_age_group(x) if x else "adult"
)

# Output the predictions for rows labeled "unspecified"
unspecified_rows = data[data["age_group_hint"] == "unspecified"]

# Print only these rows
print(unspecified_rows[["name", "topic_keywords", "predicted_age_group"]])

toys_df['age_group_hint'] = unspecified_rows['age_group_hint']

plt.subplot()
toys_df['predicted_age_group'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Age Group Distribution')
plt.show()

print("Dataset Info:")
print("-" * 40)
print(toys_df.info())

print("\nSummary Statistics:")
print("-" * 40)
print(toys_df.describe())


print(toys_df.isnull().sum())


print(toys_df.info())
print(toys_df.describe())


# 3. Age Group Distribution

toys_df['predicted_age_group'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Age Group Distribution')
plt.show()

# 1. Complexity Score Distribution

sns.histplot(df['complexity_score'], kde=True)
plt.title('Text Complexity Distribution')
plt.show()

# 4. Correlation Heatmap

numeric_df = toys_df.select_dtypes(include=['number'])  # Select only numeric columns
correlation_matrix = numeric_df.corr()  # Calculate the correlation matrix

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            vmin=-1,  # Set minimum value of the heatmap scale
            vmax=1) 
plt.title('Feature Correlations')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric features for correlation and visualization
features = ['text_length', 'sentences_count', 'complexity_score', 'readability_score']

# Plot the distributions of these features grouped by 'predicted_age_group'
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)  # Adjust rows and columns based on the number of features
    sns.histplot(data=toys_df, x=feature, hue='predicted_age_group', bins=30, multiple="stack")
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Correlation heatmap for the selected numerical features
correlation_matrix = toys_df[features].corr()  # Correlation matrix using only numerical features
plt.figure(figsize=(10, 10))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    vmin=-1, 
    vmax=1
)
plt.title('Correlation Heatmap for Selected Features')
plt.show()

