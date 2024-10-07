import numpy as np


def digit_counts(passwords, norm=False):
    digit_counts = np.array([sum(char.isdigit() for char in pwd) for pwd in passwords])

    if norm:
        lengths = np.array([len(pwd) for pwd in passwords])
        return digit_counts / lengths  # Normalize by password length
    else:
        return digit_counts


passwords = ["password123", "secure", "123456", "mypassword"]

# Conteggio delle cifre normalizzato
normalized_counts = digit_counts(passwords, norm=True)
print(normalized_counts)

# Conteggio delle cifre non normalizzato
counts = digit_counts(passwords)
print(counts)