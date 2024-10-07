import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def count_strengths(np_data):
    """
    Counts the occurrences of each strength value in a given NumPy array.
    :param np_data: A NumPy array containing the data
    :return: A dictionary where the keys are the strength values (0, 1, 2) and the values are their corresponding counts
    """
    # Extract the strength column as integers
    strengths = np_data[:, 1].astype(int)  # cast  to integers to ensure correct data types

    # Count occurrences of each strength value
    count_0 = np.sum(strengths == 0)
    count_1 = np.sum(strengths == 1)
    count_2 = np.sum(strengths == 2)

    # Return the results as a dictionary
    return {
        "Strength 0": count_0,
        "Strength 1": count_1,
        "Strength 2": count_2
    }


def plot_strength_distribution(counts):
    """
    Plots a bar chart showing the distribution of strength levels in a dataset.
    :param counts:  A dictionary containing the counts of each strength level (0, 1, 2).
    :return: None. This function simply displays the plot.
    """

    # Extract categories and corresponding counts from the dictionary
    categories = ['Strength 0', 'Strength 1', 'Strength 2']
    values = [counts['Strength 0'], counts['Strength 1'], counts['Strength 2']]

    # Set custom colors for the bars
    colors = ['red', 'orange', 'green']

    # Create a bar chart
    plt.bar(categories, values, color=colors)

    # Add a title and axis labels
    plt.title('Password Strength Distribution')
    plt.xlabel('Strength Level')
    plt.ylabel('Number of Passwords')

    # Display the plot
    plt.show()


def load_data(file_path):
    """
    Read the CSV file, and skip bad formated lines
    :param file_path: The path to the CSV file.
    :return: A Pandas DataFrame containing the data from the CSV file, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading the file: {e}")
        return None
    return df


def convert_DataFrame_to_Numpy(DataFrame):
    """
    Convert the DataFrame to numpy array
    :param DataFrame: Dataset in DataFrame format
    :return: the Dataset in numpy array format
    """
    np_data = DataFrame.to_numpy()
    return np_data
#