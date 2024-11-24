# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load dataset
heart = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split dataset into features (X) and target variable (y)
X = heart.drop(columns=['DEATH_EVENT'])  # Features
y = heart['DEATH_EVENT']  # Target

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Visualize the distribution of the target variable using a bar chart
sns.countplot(x=y, hue=y.astype(str), palette={'0': 'green', '1': 'red'}, legend=False)
plt.title("Distribution of Death Event Outcomes")
plt.xlabel("Death Event")
plt.ylabel("Count")
plt.xticks([0, 1], ['No Death', 'Death'])  # Update x-axis labels
plt.show()

# Check for data imbalance
positive_outcomes, negative_outcomes = y.value_counts()
total_samples = y.count()
print("\n----------Checking for Data Imbalance------------")
print(f'Number of Positive Outcomes (Death): {positive_outcomes}')
print(f'Percentage of Positive Outcomes (Death): {round((positive_outcomes / total_samples) * 100, 2)}%')
print(f'Number of Negative Outcomes (No Death): {negative_outcomes}')
print(f'Percentage of Negative Outcomes (No Death): {round((negative_outcomes / total_samples) * 100, 2)}%\n')

# Split the data into features and target variable
X = heart.drop(columns=['DEATH_EVENT'])  # Features
y = heart['DEATH_EVENT']  # Target variable

# Correlation Analysis
print("Calculating correlations with target and features...\n")

# Calculate the correlation matrix for features
correlation_matrix = X.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()

# Show the correlation of each feature with the target variable
correlation_with_target = X.corrwith(y).sort_values(ascending=False)
print("Correlation of features with DEATH_EVENT:")
print(correlation_with_target)

# Explanation of the correlation with DEATH_EVENT
print("\nExplanation of Correlations with 'DEATH_EVENT':\n")
for feature, correlation in correlation_with_target.items():
    if correlation > 0.5:
        print(f"{feature}: Strong positive correlation (Feature increases, likelihood of death increases).")
    elif correlation < -0.5:
        print(f"{feature}: Strong negative correlation (Feature increases, likelihood of death decreases).")
    elif correlation > 0:
        print(f"{feature}: Weak positive correlation (Feature slightly increases, likelihood of death slightly increases).")
    elif correlation < 0:
        print(f"{feature}: Weak negative correlation (Feature slightly increases, likelihood of death slightly decreases).")
    else:
        print(f"{feature}: No significant correlation with DEATH_EVENT.")

# Define metric calculation functions
def calc_metrics(conf_matrix, y_test, predicted):
    TP, FN = conf_matrix[0][0], conf_matrix[0][1]
    FP, TN = conf_matrix[1][0], conf_matrix[1][1]
    Accuracy = accuracy_score(y_test, predicted)
    Precision = precision_score(y_test, predicted)
    F1_measure = f1_score(y_test, predicted)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    metrics = [TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure, Accuracy]
    return metrics

def get_metrics(model, X_train, X_test, y_train, y_test, is_gru=False):
    if is_gru:
        # Convert DataFrame to NumPy array and reshape
        X_train_array = X_train.to_numpy()
        X_test_array = X_test.to_numpy()
        X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1], 1)
        X_test_reshaped = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1], 1)
        model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
        predicted = (model.predict(X_test_reshaped) > 0.5).astype(int)
    else:
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted, labels=[1, 0])
    metrics = calc_metrics(matrix, y_test, predicted)
    return metrics

def cross_validate_and_evaluate(X_train, y_train):
    cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

    results = {
        'RF': [],
        'Naive Bayes': [],
        'GRU': []
    }

    for iter_num, (train_index, test_index) in enumerate(cv_stratified.split(X_train, y_train), start=1):
        features_train, features_test = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        labels_train, labels_test = y_train[train_index], y_train[test_index]
        
        # Random Forest Model
        rf_model = RandomForestClassifier(min_samples_split=2, n_estimators=100)
        rf_metrics = get_metrics(rf_model, features_train, features_test, labels_train, labels_test)
        results['RF'].append(rf_metrics)
        
        # Naive Bayes Model
        nb_model = GaussianNB()
        nb_metrics = get_metrics(nb_model, features_train, features_test, labels_train, labels_test)
        results['Naive Bayes'].append(nb_metrics)
        
        # GRU Model
        gru_model = Sequential()
        gru_model.add(GRU(64, input_shape=(features_train.shape[1], 1), return_sequences=False))
        gru_model.add(Dropout(0.2))
        gru_model.add(Dense(1, activation='sigmoid'))
        gru_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        gru_metrics = get_metrics(gru_model, features_train, features_test, labels_train, labels_test, is_gru=True)
        results['GRU'].append(gru_metrics)

    # Create DataFrames for each model with iterations as rows and metrics as columns
    metric_names = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision', 'F1_measure', 'Accuracy']
    rf_metrics_df = pd.DataFrame(results['RF'], columns=metric_names)
    nb_metrics_df = pd.DataFrame(results['Naive Bayes'], columns=metric_names)
    gru_metrics_df = pd.DataFrame(results['GRU'], columns=metric_names)

    print("\nRandom Forest Metrics Across Iterations:")
    print(rf_metrics_df.round(2))
    print("\nNaive Bayes Metrics Across Iterations:")
    print(nb_metrics_df.round(2))
    print("\nGRU Metrics Across Iterations:")
    print(gru_metrics_df.round(2))

    avg_metrics_rf = rf_metrics_df.mean()
    avg_metrics_nb = nb_metrics_df.mean()
    avg_metrics_gru = gru_metrics_df.mean()

    avg_performance_df = pd.DataFrame({
        'RF': avg_metrics_rf,
        'Naive Bayes': avg_metrics_nb,
        'GRU': avg_metrics_gru
    }).T

    print("\nAverage Performance Across 10 Iterations (Cross-Validation):")
    print(avg_performance_df.round(2))

# Run the cross-validation and evaluation using the heart dataset
cross_validate_and_evaluate(X, y)