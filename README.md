# 🧩 Task 8: K-Means Clustering – AI & ML Internship

## 📌 Objective

Perform **unsupervised learning** using **K-Means clustering** to segment data into meaningful groups.

## 🛠️ Tools & Libraries

* Python 3.x
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Pickle (for saving the model)

## 📂 Dataset

**Mall Customer Segmentation Dataset** (Mall_Customers.csv)

Features used:

* Annual Income (k$)
* Spending Score (1–100)
* (Optional: Age and others for extended analysis)

## 📖 Steps Implemented

1. **Data Loading & Exploration**

   * Loaded dataset using Pandas.
   * Visualized data distributions and relationships.

2. **Feature Selection & Scaling**

   * Selected `Annual Income` & `Spending Score`.
   * Standardized data using `StandardScaler`.

3. **Elbow Method**

   * Plotted **inertia vs k** to find the optimal number of clusters.
   * Observed the elbow at **k ≈ 5**.

4. **Silhouette Score Analysis**
   Tested different values of k:

   | k     | Silhouette Score |
   | ----- | ---------------- |
   | 2     | 0.3973           |
   | 3     | 0.4666           |
   | 4     | 0.4943           |
   | **5** | **0.5547 ✅**     |
   | 6     | 0.5138           |
   | 7     | 0.5020           |
   | 8     | 0.4550           |
   | 9     | 0.4567           |

   ✅ Best clustering achieved at **k = 5**.

5. **Model Training & Visualization**

   * Trained **K-Means with k=5**.
   * Visualized clusters with scatter plots & centroids.

6. **Model Saving (Pickle)**

   * Saved trained model as `kmeans_model.pkl`.
   * Example usage:

   ```python
   import pickle
   import pandas as pd

   # Load model
   with open("kmeans_model.pkl", "rb") as file:
       model = pickle.load(file)

   # Predict clusters on new data
   new_clusters = model.predict(X_scaled)
   print(new_clusters)
   ```

## 📊 Results & Insights

* Dataset naturally forms **5 distinct customer segments**.
* Clusters identified groups like **low income–high spenders**, **high income–low spenders**, etc.
* Silhouette Score (~0.55) indicates a **reasonable cluster separation**.

## 📷 Screenshots (Add in Repo)

* Elbow Method Plot
* Cluster Visualization (colored scatter plot)

## 📌 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/kmeans-clustering-task.git
   cd kmeans-clustering-task
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:

   ```bash
   jupyter notebook kmeans_clustering.ipynb
   ```

## 🙋 Interview Questions Covered

1. How does K-Means clustering work?
2. What is the Elbow method?
3. What are the limitations of K-Means?
4. How does initialization affect results?
5. What is inertia in K-Means?
6. What is Silhouette Score?
7. How do you choose the right number of clusters?
8. Difference between clustering and classification?

## 📌 Files in Repo

* `kmeans_clustering.ipynb` → Jupyter Notebook with full code
* `Mall_Customers.csv` → Dataset
* `kmeans_model.pkl` → Saved trained model
* `README.md` → Project documentation

✨ **Author**: Chakresh Kumar Vulli
📅 Internship Project – Task 8
