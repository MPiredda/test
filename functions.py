import pandas as pd
import numpy as np
from scipy.stats import entropy

def password_length(password):
    lengths = np.array([len(str(psw)) for psw in password])
    return lengths


def password_entropy(password):
    entropy_values = np.array([entropy([ord(char) for char in str(psw)], base=2) for psw in password], dtype=np.float64)
    return entropy_values


def password_complessity(passwords):

    complessity = np.array([str(psw) for psw in passwords])
    scores = np.zeros(len(passwords), dtype=int)

    for i, password in enumerate(complessity):
        score = 0

        if any(char.islower() for char in password):
            score += 1
        if any(char.isupper() for char in password):
            score += 1
        if any(char.isdigit() for char in password):
            score += 1
        if any(char.isascii() and not char.isalnum() for char in password):
            score += 1

        scores[i] = score

        # str.istitle() vedere se aggiungerlo

    return scores

