import pandas as pd
import numpy as np

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

# Normalization helper function (min-max normalization)
# Normalization based on password length
def normalized_digit_counts(passwords):
    digit_counts = np.array([sum(char.isdigit() for char in pwd) for pwd in passwords])
    lengths = np.array([len(pwd) for pwd in passwords])
    return digit_counts / lengths  # Normalize by password length

def normalized_lowercase_counts(passwords):
    lowercase_counts = np.array([sum(char.islower() for char in pwd) for pwd in passwords])
    lengths = np.array([len(pwd) for pwd in passwords])
    return lowercase_counts / lengths  # Normalize by password length

def normalized_uppercase_counts(passwords):
    uppercase_counts = np.array([sum(char.isupper() for char in pwd) for pwd in passwords])
    lengths = np.array([len(pwd) for pwd in passwords])
    return uppercase_counts / lengths  # Normalize by password length

def normalized_special_char_counts(passwords):
    special_char_counts = np.array([sum(not char.isalnum() for char in pwd) for pwd in passwords])
    lengths = np.array([len(pwd) for pwd in passwords])
    return special_char_counts / lengths  # Normalize by password length

def normalized_unique_char_counts(passwords):
    unique_char_counts = np.array([len(set(pwd)) for pwd in passwords])
    lengths = np.array([len(pwd) for pwd in passwords])
    return unique_char_counts / lengths  # Normalize by password length

# Normalize the length of the password by the maximum length in the dataset
def normalized_password_lengths(passwords):
    lengths = np.array([len(pwd) for pwd in passwords])
    max_length = np.max(lengths)  # Find the maximum length among all passwords
    return lengths / max_length  # Normalize by the maximum password length


# Function that calculates the normalized features for each password and returns a numpy array
def calculate_normalized_password_features(passwords):
    digit_counts = normalized_digit_counts(passwords)
    lowercase_counts = normalized_lowercase_counts(passwords)
    uppercase_counts = normalized_uppercase_counts(passwords)
    special_char_counts = normalized_special_char_counts(passwords)
    unique_char_counts = normalized_unique_char_counts(passwords)
    total_lengths = normalized_password_lengths(passwords)

    # Concatenate the normalized features into a single array
    features = np.column_stack((digit_counts, lowercase_counts, uppercase_counts, special_char_counts,
                                unique_char_counts, total_lengths))

    return features

# Example usage
dataSet = "data.csv"
dataSet_Featured = ".csv"

df, passwords = read_password_file(dataSet)
features = calculate_normalized_password_features(passwords)
save_featured_dataset(df, features, dataSet_Featured)
#