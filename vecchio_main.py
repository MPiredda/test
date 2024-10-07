import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.utils import shuffle
import joblib
import time
import matplotlib.pyplot as plt


# Function to load, shuffle and prepare the dataset
def load_and_prepare_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Shuffle the dataset
    df = shuffle(df, random_state=42)

    # Extract features and labels
    X = df[['Digit Count', 'Lowercase Count', 'Uppercase Count', 'Special Char Count', 'Unique Char Count',
            'Total Length']].values
    y = df['strength'].values  # Assuming the complexity level is in a column named 'strength'

    return X, y


# Function to train the MLP model
def train_mlp(X_train, y_train):
    start_time = time.time()  # Start timing

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train the MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=400, random_state=32)
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


# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test, model_name, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)

    # Predict probabilities for the test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1 (positive class)

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)  # Assume binary classification with pos_label=1
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
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
    # Load and prepare the data
    file_path = 'dataset_features.csv'
    X, y = load_and_prepare_data(file_path)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the MLP model and Random Forest model
    mlp, scaler, training_time = train_mlp(X_train, y_train)
    rf, training_time = train_rf(X_train, y_train)

    # Evaluate the MLP model and Random Forest model
    evaluate_mlp(mlp, scaler, X_test, y_test)
    evaluate_rf(rf, X_test, y_test)

    # Plot ROC curve for MLP and Random Forest
    plot_roc_curve(mlp, X_test, y_test, model_name="MLP", scaler=scaler)
    plot_roc_curve(rf, X_test, y_test, model_name="Random Forest")


if __name__ == '__main__':
    main()
#