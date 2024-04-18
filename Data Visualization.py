# Importing the Libraries which we are using for Data Visualization


import pandas as pd   # pandas is used for data manipulation and analysis
import matplotlib.pyplot as plt   # matplotlib is used to create static, interactive, and animated visualizations 
import seaborn as sns   # seaborn is provides a high-level interface for drawing attractive statistical graphics
from sklearn.cluster import KMeans   # machine learning library 

# Load the data
df = pd.read_csv('transaction_data.csv')

# Data preprocessing

#check for null values in the data
df.isna().sum()   

#check for any duplicatge values in the data
df.duplicated().sum()

#dropping the duplicate values from the data
df.drop_duplicate()

#to view the no.of rows & columns
df.shape

#to view the data type 
df.info()


# Exploratory Data Analysis (EDA)
# Sales analysis
sales_by_month = transactions.groupby('Month')['Sales'].sum()
plt.figure(figsize=(10, 6))
sns.lineplot(data=sales_by_month, marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Customer segmentation
# Feature selection and scaling
X = df[['TotalSpent', 'Transactions']]
# Normalize data if necessary

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize customer segmentation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=transactions, x='TotalSpent', y='Transactions', hue='Cluster', palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Total Amount Spent')
plt.ylabel('Number of Transactions')
plt.legend(title='Cluster')
plt.show()
