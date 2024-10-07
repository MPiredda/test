import numpy as np

def count_digits(passwords, lengths, norm=False):
    digit_counts = np.array([sum(char.isdigit() for char in pwd) for pwd in passwords])
    if norm:
        return digit_counts / lengths
    else:
        return digit_counts

def count_lowercase_letters(passwords, lengths, norm=False):
    lowercase_counts = np.array([sum(char.islower() for char in pwd) for pwd in passwords])
    if norm:
        return lowercase_counts / lengths
    else:
        return lowercase_counts

def count_uppercase_letters(passwords, lengths, norm=False):
    uppercase_counts = np.array([sum(char.isupper() for char in pwd) for pwd in passwords])
    if norm:
        return uppercase_counts / lengths
    else:
        return uppercase_counts

def count_special_characters(passwords, lengths, norm=False):
    special_char_counts = np.array([sum(not char.isalnum() for char in pwd) for pwd in passwords])
    if norm:
        return special_char_counts / lengths
    else:
        return special_char_counts

def count_unique_characters(passwords, lengths, norm=False):
    unique_char_counts = np.array([len(set(pwd)) for pwd in passwords])
    if norm:
        return unique_char_counts / lengths
    else:
        return unique_char_counts

def calculate_password_length(passwords):
    lengths = np.array([len(pwd) for pwd in passwords])
    return lengths

def normalized_password_lengths(passwords):
    lengths = np.array([len(pwd) for pwd in passwords])
    max_length = np.max(lengths)
    return lengths / max_length

def calculate_password_features(passwords, norm=False):
    lengths = calculate_password_length(passwords)
    digit_counts = count_digits(passwords, lengths, norm)
    lowercase_counts = count_lowercase_letters(passwords, lengths, norm)
    uppercase_counts = count_uppercase_letters(passwords, lengths, norm)
    special_char_counts = count_special_characters(passwords, lengths, norm)
    unique_char_counts = count_unique_characters(passwords, lengths, norm)
    if norm:
        lengths = normalized_password_lengths(passwords)


    features = np.column_stack((digit_counts, lowercase_counts, uppercase_counts, special_char_counts,
                                unique_char_counts, lengths))
    return features