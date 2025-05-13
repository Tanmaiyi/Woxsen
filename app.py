import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('analyze', filename=file.filename))
    return render_template('index.html')

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)

    # Clear previous images
    for f in os.listdir(STATIC_FOLDER):
        if f.endswith('.png'):
            os.remove(os.path.join(STATIC_FOLDER, f))

    overview = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'describe': df.describe().round(2),
        'missing_values': df.isnull().sum().to_dict()
    }

    images = []
    
    # 1. Data Distribution Visualizations
    if 'Age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'], bins=20, kde=True, color='royalblue')
        plt.title('Age Distribution', fontsize=14)
        plt.xlabel('Age')
        plt.ylabel('Count')
        fname = 'age_distribution.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Age Distribution'})

    if 'Gender' in df.columns:
        plt.figure(figsize=(8, 6))
        gender_counts = df['Gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=['lightcoral', 'lightskyblue'], startangle=90)
        plt.title('Gender Distribution', fontsize=14)
        fname = 'gender_distribution.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Gender Distribution'})

    if 'Gender' in df.columns and 'Spending Score (1-100)' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df, 
                   palette={'Male': 'lightblue', 'Female': 'lightpink'})
        plt.title('Spending Score by Gender', fontsize=14)
        fname = 'spending_score_by_gender.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Spending Score by Gender'})

    # 2. Correlation Analysis
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in df.columns for col in features):
        plt.figure(figsize=(10, 8))
        corr = df[features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix', fontsize=14)
        fname = 'correlation_matrix.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Feature Correlation'})

        # Pairplot
        plt.figure(figsize=(10, 8))
        sns.pairplot(df[features], diag_kind='kde')
        plt.suptitle('Feature Relationships', y=1.02, fontsize=14)
        fname = 'feature_relationships.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Feature Relationships'})

    # 3. K-Means Clustering Analysis
    if all(col in df.columns for col in features):
        X = df[features]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow Method
        plt.figure(figsize=(10, 6))
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='royalblue')
        plt.title('Elbow Method for Optimal Cluster Number', fontsize=14)
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.grid(True)
        fname = 'elbow_method.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Elbow Method Analysis'})

        # K-Means Clustering with 5 clusters (as per elbow method)
        optimal_clusters = 5
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X_scaled)
        df['Cluster'] = y_kmeans
        
        # Cluster Visualizations
        cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # Age vs Income
        plt.figure(figsize=(12, 8))
        for i in range(optimal_clusters):
            plt.scatter(X.iloc[y_kmeans == i, 0], X.iloc[y_kmeans == i, 1], 
                        s=100, c=cluster_colors[i], label=f'Cluster {i+1}')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    s=300, c='yellow', label='Centroids', marker='*', edgecolor='black')
        plt.title('Customer Segments: Age vs Annual Income', fontsize=14)
        plt.xlabel('Age')
        plt.ylabel('Annual Income (k$)')
        plt.legend()
        plt.grid(True)
        fname = 'age_vs_income_clusters.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Age vs Income Clusters'})

        # Income vs Spending Score
        plt.figure(figsize=(12, 8))
        for i in range(optimal_clusters):
            plt.scatter(X.iloc[y_kmeans == i, 1], X.iloc[y_kmeans == i, 2], 
                        s=100, c=cluster_colors[i], label=f'Cluster {i+1}')
        plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
                    s=300, c='yellow', label='Centroids', marker='*', edgecolor='black')
        plt.title('Customer Segments: Income vs Spending Score', fontsize=14)
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.grid(True)
        fname = 'income_vs_spending_clusters.png'
        plt.savefig(os.path.join(STATIC_FOLDER, fname), bbox_inches='tight')
        plt.close()
        images.append({'name': fname, 'title': 'Income vs Spending Clusters'})

        # Cluster Profile Analysis
        cluster_profiles = df.groupby('Cluster')[features].mean().round(2)
        overview['cluster_profiles'] = cluster_profiles.to_dict('index')

    return render_template('analysis.html', overview=overview, images=images)

if __name__ == "__main__":
    app.run(debug=True)