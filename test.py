import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Function to load, shuffle and prepare the dataset
def load_and_prepare_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Shuffle the dataset
    df = shuffle(df, random_state=42)

    # Extract features and labels
    X = df[['Digit Count', 'Lowercase Count', 'Uppercase Count', 'Special Char Count', 'Unique Char Count', 'Total Length']].values
    y = df['strength'].values
    return X, y

# Function to train the MLP model
def train_mlp(X_train, y_train):
    start_time = time.time()  # Start timing

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train the MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=400, random_state=32)
    mlp.fit(X_train_scaled, y_train)

    end_time = time.time()  # End timing
    training_time = end_time - start_time  # Calculate training time

    print(f"MLP Training Time: {training_time:.2f} seconds")

    # Save the trained model and scaler
    joblib.dump(mlp, 'mlp_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    print("MLP model and scaler saved.")

    return mlp, scaler, training_time

# Function to evaluate the MLP model
def evaluate_mlp(mlp, scaler, X_test, y_test):
    # Scale the test set using the saved scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict and evaluate the model
    y_pred = mlp.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("MLP Classifier Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

# Function to train the Random Forest model
def train_rf(X_train, y_train):
    start_time = time.time()  # Start timing

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(n_estimators=80, random_state=32)
    rf.fit(X_train, y_train)
    end_time = time.time()  # End timing
    training_time = end_time - start_time  # Calculate training time

    print(f"Random Forest Training Time: {training_time:.2f} seconds")

    # Save the trained model
    joblib.dump(rf, 'rf_model.joblib')
    print("Random Forest model saved.")

    return rf, training_time

# Function to evaluate the Random Forest model
def evaluate_rf(rf, X_test, y_test):
    # Predict and evaluate the model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Random Forest Classifier Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

# Function to train the Deep Neural Network (DNN) model
def train_dnn(X_train, y_train):
    start_time = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    dnn = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=32)
    dnn.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"DNN Training Time: {training_time:.2f} seconds")
    joblib.dump(dnn, 'dnn_model.joblib')
    joblib.dump(scaler, 'dnn_scaler.joblib')
    print("DNN model and scaler saved.")
    return dnn, scaler, training_time

# Function to evaluate the Deep Neural Network (DNN) model
def evaluate_dnn(dnn, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = dnn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("DNN Classifier Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

# Function to train the Gradient Boosting (XGBoost) model
def train_xgboost(X_train, y_train):
    start_time = time.time()
    xgb = XGBClassifier(n_estimators=100, random_state=32, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"XGBoost Training Time: {training_time:.2f} seconds")
    joblib.dump(xgb, 'xgb_model.joblib')
    print("XGBoost model saved.")
    return xgb, training_time

# Function to evaluate the XGBoost model
def evaluate_xgboost(xgb, X_test, y_test):
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("XGBoost Classifier Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test, model_name, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    y_prob = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(y_test_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Main function
def main():
    file_path = 'dataset_features.csv'
    X, y = load_and_prepare_data(file_path)

    # First split: training (70%) and test+validation (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

    # Second split: validation (10% of the original dataset) and test (20% of the original dataset) from the temporary split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)  # 2/3 of 30% = 20% of the original dataset

    # Train models
    mlp, scaler_mlp, _ = train_mlp(X_train, y_train)
    rf, _ = train_rf(X_train, y_train)
    dnn, scaler_dnn, _ = train_dnn(X_train, y_train)
    xgb, _ = train_xgboost(X_train, y_train)

    # Evaluate models on the validation set
    print("Evaluating on validation set:")
    evaluate_mlp(mlp, scaler_mlp, X_val, y_val)
    evaluate_rf(rf, X_val, y_val)
    evaluate_dnn(dnn, scaler_dnn, X_val, y_val)
    evaluate_xgboost(xgb, X_val, y_val)

    # Evaluate models on the test set
    print("Evaluating on test set:")
    evaluate_mlp(mlp, scaler_mlp, X_test, y_test)
    evaluate_rf(rf, X_test, y_test)
    evaluate_dnn(dnn, scaler_dnn, X_test, y_test)
    evaluate_xgboost(xgb, X_test, y_test)

    # Plot ROC curves for the test set
    print("Plotting ROC curves for test set:")
    plot_roc_curve(mlp, X_test, y_test, model_name="MLP", scaler=scaler_mlp)
    plot_roc_curve(rf, X_test, y_test, model_name="Random Forest")
    plot_roc_curve(dnn, X_test, y_test, model_name="DNN", scaler=scaler_dnn)
    plot_roc_curve(xgb, X_test, y_test, model_name="XGBoost")

if __name__ == '__main__':
    main()