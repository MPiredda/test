import pandas as pd
import numpy as np


# Functions to extract various features from passwords:
# These functions count digits, lowercase letters, uppercase letters, special characters,
# unique characters, and calculate the total length for each password in a given list.

def count_digits(passwords):
    digit_counts = np.array([sum(char.isdigit() for char in pwd) for pwd in passwords])
    return digit_counts


def count_lowercase_letters(passwords):
    lowercase_counts = np.array([sum(char.islower() for char in pwd) for pwd in passwords])
    return lowercase_counts


def count_uppercase_letters(passwords):
    uppercase_counts = np.array([sum(char.isupper() for char in pwd) for pwd in passwords])
    return uppercase_counts


def count_special_characters(passwords):
    special_char_counts = np.array([sum(not char.isalnum() for char in pwd) for pwd in passwords])
    return special_char_counts


def count_unique_characters(passwords):
    unique_char_counts = np.array([len(set(pwd)) for pwd in passwords])
    return unique_char_counts


def calculate_password_length(passwords):
    lengths = np.array([len(pwd) for pwd in passwords])
    return lengths


# Function that calculates the features for each password and returns a numpy array
def calculate_password_features(passwords):
    digit_counts = count_digits(passwords)
    lowercase_counts = count_lowercase_letters(passwords)
    uppercase_counts = count_uppercase_letters(passwords)
    special_char_counts = count_special_characters(passwords)
    unique_char_counts = count_unique_characters(passwords)
    total_lengths = calculate_password_length(passwords)

    # Concatenate the features into a single array
    features = np.column_stack((digit_counts, lowercase_counts, uppercase_counts, special_char_counts,
                                unique_char_counts, total_lengths))
    return features


# Function that reads the CSV file and extracts the passwords
def read_password_file(input_csv_path):
    # Reads the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path, on_bad_lines='skip')

    # Ensures the password column is treated as a string and handles missing values
    df['password'] = df['password'].astype(str).fillna('')

    # Extracts the password column as a NumPy array
    passwords = df['password'].to_numpy()

    return df, passwords


# Function that saves the dataset with extracted features to a CSV file
def save_featured_dataset(df, features, output_csv_path):
    # Concatenate the original dataset with the new features
    extended_dataset = np.column_stack((df.to_numpy(), features))

    # Create a new DataFrame with extended column names
    extended_df = pd.DataFrame(extended_dataset, columns=list(df.columns) + [
        'Digit Count', 'Lowercase Count', 'Uppercase Count', 'Special Char Count', 'Unique Char Count', 'Total Length'
    ])

    # Save the extended dataset to a CSV file
    extended_df.to_csv(output_csv_path, index=False)
    print(f"Featured dataset saved to {output_csv_path}")


# Example usage
dataSet = "data.csv"
dataSet_Featured = ".csv"  # Changed filename to reflect the "features" naming
df, passwords = read_password_file(dataSet)
features = calculate_password_features(passwords)
save_featured_dataset(df, features, dataSet_Featured)
#