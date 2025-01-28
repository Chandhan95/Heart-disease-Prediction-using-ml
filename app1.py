from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Step 1: Load your dataset (replace 'your_actual_dataset.csv' with the actual dataset file name or path)
# If the file is in the same directory as your script, just use the file name like 'heart_disease_data.csv'
data = pd.read_csv('heart.csv')
 # Update the file name here

X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column is the target variable

# Ensure the dataset has exactly 13 features
if X.shape[1] != 13:
    raise ValueError("The input dataset must have exactly 13 features.")

# Step 2: Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

# Step 4: Save the trained model
joblib.dump(knn_model, 'Heart-Prediction-KNN-Classifier.joblib')

# Step 5: Load the saved model
loaded_knn_model = joblib.load('Heart-Prediction-KNN-Classifier.joblib')

# Example input data for prediction (ensure these values match the 13 features)
input_data = [[60, 1, 5, 2003, 2003, 856.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

# Step 6: Make predictions using the loaded model
predictions = loaded_knn_model.predict(input_data)

# Output the prediction
print("Heart Prediction (0 or 1):", predictions)
