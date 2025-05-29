📌 Project Overview The goal of this project was to:

Understand the relationship between different house features and their price.

Build and evaluate machine learning models for regression.

Learn and apply feature preprocessing, model optimization, and evaluation techniques.

📂 Dataset Source: Kaggle Housing Price Dataset (Uploaded)

Total records: 545 entries

Features include both numerical and categorical data (like location, furnishing status, presence of amenities, etc.)

🧠 Steps Followed Data Preprocessing

Loaded dataset using Pandas.

Assigned the target variable (house price) to y and features to X.

Encoded categorical variables using Label Encoding and One Hot Encoding.

Handled the dummy variable trap by removing one dummy column for each categorical variable.

Feature Selection

Used Backward Elimination (based on p-values) with StatsModels to select significant features affecting the house price.

Train-Test Split

Split the dataset into 80% training and 20% testing using train_test_split.

Feature Scaling

Applied StandardScaler to handle feature ranges and outliers.

Model Building

Built and evaluated the following models:

🔹 Multiple Linear Regression

🌳 Decision Tree Regressor

🌲 Random Forest Regressor

Model Evaluation

Used metrics to evaluate model performance:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Visualized Actual vs Predicted Price for both Training and Testing sets.

📊 Results & Visualizations Below are some sample visualizations showing actual vs predicted prices: Figure 2025-04-14 220226 (5) Figure 2025-04-14 220226 (4) Figure 2025-04-14 220226 (3) Figure 2025-04-14 220226 (2) Figure 2025-04-14 220226 (1) Figure 2025-04-14 220226 (0)

🔍 Observations Despite using advanced models like Random Forest, the R² score was relatively low.

This might be due to:

Limited size or quality of the dataset.

Missing important features (like location-specific factors).

Noise in data or outliers not captured fully.

💡 Learnings Learned how to preprocess and encode real-world data.

Understood the importance of feature selection and avoiding data leakage.

Gained hands-on experience with model evaluation metrics.

Realized the importance of visual analysis alongside numerical metrics.
