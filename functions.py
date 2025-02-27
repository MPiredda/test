import pandas as pd
import numpy as np
import csv



def save_to_file(data, file_name):
    """
    Save passwords to a file, one password per line.

    Args:
    - passwords (list or numpy array): List or array of passwords.
    - file_name (str): The name of the file to save passwords.
    """
    # Open the file with utf-8 encoding
    with open(file_name, 'w', encoding='utf-8') as file:
        for data in data:
            file.write(data + '\n')


def clean_and_filter_csv(input_file, output_file):
    """
    Cleans and filters a CSV file by removing malformed rows and ensuring the first column
    has between 4 and 30 characters.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output cleaned CSV file.
    """

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        cleaned_rows = [row for row in reader if len(row) == 2 and 4 <= len(row[0]) <= 30]

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)