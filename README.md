# Global Development Clustering Project

This project applies multiple clustering algorithms to a global development dataset containing 2704 records and 25 features. The goal is to segment countries based on socio-economic, demographic, and development indicators.

## Dataset Overview
The dataset includes features such as Birth Rate, CO2 Emissions, GDP, Health Expenditure, Energy Usage, Internet Usage, Infant Mortality, Life Expectancy, Lending Rate, and more.  
Total: 2704 rows × 25 columns.

## Data Preprocessing
- Imputed missing values (median for numeric, mode for categorical)
- Dropped features with >50% missing values
- Winsorization used for outlier handling
- Scaled and normalized numerical columns
- Encoded categorical columns
- No duplicates or datatype mismatches

## Exploratory Data Analysis (EDA)
- Univariate: Boxplots to detect skewness & outliers  
- Bivariate: Histplots & scatterplots for distribution behavior  
- Multivariate: Pairplots & correlation heatmap to study relationships  
Key observations:  
- Most numerical features are strongly positively skewed  
- Strong correlations:  
  - Life Expectancy ↔ Infant Mortality (negative)  
  - GDP ↔ Health Expenditure & Energy Usage (positive)  
  - Internet/Mobile Usage ↔ GDP (positive)

## Clustering Models Applied
Five clustering techniques were evaluated:
1. K-Means  
2. Agglomerative Clustering  
3. K-Medoids  
4. Gaussian Mixture Models (GMM)  
5. DBSCAN  

The optimal number of clusters for centroid-based methods was **k = 3** (based on Elbow Curve & Silhouette Score).

### Model Behavior Summary
- **K-Means:** 3 broad segments of countries with similar patterns  
- **Agglomerative:** Similar 3-cluster structure but hierarchy-based  
- **K-Medoids:** One dominant cluster + smaller distinct groups  
- **GMM:** Soft overlapping clusters  
- **DBSCAN:** Detected 9 natural-density clusters + noise points  

## Model Evaluation
Used **Davies–Bouldin Index** to compare model quality.  
**DBSCAN achieved the lowest DB score**, indicating the best cluster separation and stability.

## Final Model Selection
**DBSCAN** was chosen as the final model because:
- It automatically detects cluster structure  
- Handles varying densities well  
- Identifies niche groups and outliers effectively  
- Produces the most realistic segmentation of countries

### Final DBSCAN Output
- 9 meaningful clusters  
- Cluster 0 = majority dense group  
- Clusters 1–8 = distinct minority segments  
- ~47 noise/outlier points

## Deployment
A Streamlit application was built where users enter feature values and receive the corresponding cluster label predicted by the DBSCAN model.

## How to run:
1.Run the jupyter file:
final_clustering.ipynb

2.Run the app:
streamlit run app.py
