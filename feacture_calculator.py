import numpy as np

def count_digits(passwords, lengths, norm=False):
    """
    Counts the number of digits in each password.

    Args:
        passwords: A numpy array containing passwords.
        lengths: Lengths of the corresponding passwords.
        norm: A boolean indicating whether to normalize the digit counts by the password lengths.

    Returns:
        A NumPy array containing the digit counts for each password. If `norm` is True, the counts are normalized.
    """

    digit_counts = np.array([sum(char.isdigit() for char in pwd) for pwd in passwords])
    if norm:
        print (lengths)
        return digit_counts / lengths
    else:
        return digit_counts

def count_lowercase_letters(passwords, lengths, norm=False):
    """
    Counts the number of lowercase letters in each password and optionally normalizes the counts.

    Args:
        passwords: A numpy array containing passwords.
        lengths: Lengths of the corresponding passwords.
        norm: A boolean indicating whether to normalize the digit counts by the password lengths.

    :return:  A NumPy array containing the lowercase counts for each password. If `norm` is True, the counts are normalized.
    """
    lowercase_counts = np.array([sum(char.islower() for char in pwd) for pwd in passwords])
    if norm:
        return lowercase_counts / lengths
    else:
        return lowercase_counts

def count_uppercase_letters(passwords, lengths, norm=False):
    """
    Counts the number of uppercase letters in each password and optionally normalizes the counts.

    Args:
        passwords: A numpy array containing passwords.
        lengths: Lengths of the corresponding passwords.
        norm: A boolean indicating whether to normalize the digit counts by the password lengths.

    :return: A NumPy array containing counts of uppercase letters (if norm=False) for each password. If `norm` is True, the counts are normalized.
    """
    uppercase_counts = np.array([sum(char.isupper() for char in pwd) for pwd in passwords])
    if norm:
        return uppercase_counts / lengths
    else:
        return uppercase_counts

def count_special_characters(passwords, lengths, norm=False):
    """
    Counts the number of special characters (non-alphanumeric) in each password and optionally normalizes the counts.

    Args:
        passwords: A numpy array containing passwords.
        lengths: Lengths of the corresponding passwords.
        norm: A boolean indicating whether to normalize the digit counts by the password lengths.

    :return: A NumPy array containing the special character counts for each password. If `norm` is True, the counts are normalized.
    """
    special_char_counts = np.array([sum(not char.isalnum() for char in pwd) for pwd in passwords])
    if norm:
        return special_char_counts / lengths
    else:
        return special_char_counts

def count_unique_characters(passwords, lengths, norm=False):
    """
    Counts the number of unique characters in each password and optionally normalizes the counts.

    Args:
        passwords: A numpy array containing passwords.
        lengths: Lengths of the corresponding passwords.
        norm: A boolean indicating whether to normalize the digit counts by the password lengths.

    :return: A NumPy array containing the unique character counts for each password. If `norm` is True, the counts are normalized.
    """
    unique_char_counts = np.array([len(set(pwd)) for pwd in passwords])
    if norm:
        return unique_char_counts / lengths
    else:
        return unique_char_counts

def calculate_password_length(passwords):
    """
    Calculates the lengths of each password.

    Args:
        passwords: A numpy array containing passwords.

    :return: A NumPy array containing the lengths of each password.
    """
    lengths = np.array([len(pwd) for pwd in passwords])
    return lengths

def normalized_password_lengths(passwords):
    """
    Normalizes the lengths of passwords by dividing each password length by the maximum password length.
    Args:
        passwords: A numpy array containing passwords.

    :return: A NumPy array containing the normalized lengths of the passwords.
    """
    lengths = np.array([len(pwd) for pwd in passwords])
    max_length = np.max(lengths)
    return lengths / max_length

def calculate_password_features(np_dataset, norm=False):
    """
    Calculates various features for a list of passwords: digit counts,
    lowercase letter counts, uppercase letter counts, special character counts,
    unique character counts,  password lengths.
    Optionally a flag to normalizes all lists.

    Args:
        passwords: A numpy array containing passwords (strings).
        norm: A boolean indicating whether to normalize the counts by the password lengths.

    :return: A NumPy array where each row corresponds to a password and each column contains a specific feature:
        - Column 1: Digit counts
        - Column 2: Lowercase letter counts
        - Column 3: Uppercase letter counts
        - Column 4: Special character counts
        - Column 5: Unique character counts
        - Column 6: Password lengths (normalized if `norm` is True)
    """
    passwords = np_dataset[:, 0].astype(str)
    feacture = np_dataset[:, 1]

    lengths = calculate_password_length(passwords)
    digit_counts = count_digits(passwords, lengths, norm)
    lowercase_counts = count_lowercase_letters(passwords, lengths, norm)
    uppercase_counts = count_uppercase_letters(passwords, lengths, norm)
    special_char_counts = count_special_characters(passwords, lengths, norm)
    unique_char_counts = count_unique_characters(passwords, lengths, norm)
    if norm:
        lengths = normalized_password_lengths(passwords)


    features = np.column_stack((passwords,feacture,  digit_counts, lowercase_counts, uppercase_counts, special_char_counts,
                                unique_char_counts, lengths))
    return features

#