# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset 1: IMDB Dataset
data_imdb = pd.read_csv(r"C:\Users\venne\Downloads\sentiment_analysis-master\sentiment_analysis-master\Datasets\imdb_labelled.txt", sep="\t", header=None)
data_imdb.columns = ['text', 'sentiment']

# Dataset 2: Yelp Dataset
data_yelp = pd.read_csv(r"C:\Users\venne\Downloads\sentiment_analysis-master\sentiment_analysis-master\Datasets\yelp_labelled.txt", sep="\t", header=None)
data_yelp.columns = ['text', 'sentiment']

# Dataset 3: Amazon Cells Dataset
data_amazon = pd.read_csv(r"C:\Users\venne\Downloads\sentiment_analysis-master\sentiment_analysis-master\Datasets\amazon_cells_labelled.txt", sep="\t", header=None)
data_amazon.columns = ['text', 'sentiment']

# Combine all datasets into one
data_all = pd.concat([data_imdb, data_yelp, data_amazon], axis=0).reset_index(drop=True)

# Step 1: Preprocess the Data
X = data_all['text']  # Text data
y = data_all['sentiment']  # Sentiment labels

# Convert labels to numeric values (0 for negative, 1 for positive)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 2: Split the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Feature Extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train Models and Evaluate Performance

# Initialize classifiers
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine (SVM)": SVC(probability=True),
    "Artificial Neural Network (ANN)": MLPClassifier()
}

results = {}
accuracies = []
r2_scores = []
mae_scores = []
mse_scores = []
auc_scores = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    # R² score (only for models with regression-type behavior, e.g., MLPClassifier can be used in regression mode)
    try:
        r2 = r2_score(y_test, y_pred)
    except:
        r2 = np.nan
    r2_scores.append(r2)
    
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    
    # AUC - only for binary classification
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:, 1])
    except:
        auc = np.nan
    auc_scores.append(auc)

    report = classification_report(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")
    print(f"Classification Report for {model_name}:\n{report}")

    # Store the results
    results[model_name] = {
        "accuracy": accuracy,
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "auc": auc,
        "classification_report": report,
    }

# Step 5: Plot Model Comparison - Performance Metrics

# Create subplots for all models' metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy plot
axes[0, 0].bar(models.keys(), accuracies, color='skyblue')
axes[0, 0].set_title("Model Accuracy")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_ylim(0, 1)

# R² plot
axes[0, 1].bar(models.keys(), r2_scores, color='orange')
axes[0, 1].set_title("R² Score")
axes[0, 1].set_ylabel("R²")
axes[0, 1].set_ylim(0, 1)

# MAE plot
axes[1, 0].bar(models.keys(), mae_scores, color='lightgreen')
axes[1, 0].set_title("Mean Absolute Error (MAE)")
axes[1, 0].set_ylabel("MAE")
axes[1, 0].set_ylim(0, max(mae_scores) + 0.1)

# MSE plot
axes[1, 1].bar(models.keys(), mse_scores, color='salmon')
axes[1, 1].set_title("Mean Squared Error (MSE)")
axes[1, 1].set_ylabel("MSE")
axes[1, 1].set_ylim(0, max(mse_scores) + 0.1)

plt.tight_layout()
plt.show()

# Step 6: Select the Best Model Based on Accuracy (or other metrics)
best_model_name = max(results, key=lambda k: results[k]['accuracy'])  # Choose the model with the highest accuracy
best_model = models[best_model_name]

# Save the selected model for later use
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"\nThe selected model for prediction is: {best_model_name}")