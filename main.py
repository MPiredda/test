import numpy as np

import dataset_functions as df
import feacture_calculator as fc
def main():
    # Load dataset as Dataframe and convert it into Numpy array
    file_path = 'data.csv'
    df_dataset = df.load_data(file_path)
    np_dataset = df.convert_DataFrame_to_Numpy(df_dataset)
    passwords = np_dataset[:, 0].astype(str)

    # Count the occurrences for each strength level and Plot of the class distribution
    #result = df.count_strengths(np_dataset)
    #df.plot_strength_distribution(result)


    # calculate the password features, both normalized and unnormalized.
    features = fc.calculate_password_features(passwords, norm=False)
    eatures = fc.calculate_password_features(passwords, norm=True)

    print(features)
    print(eatures)


if __name__ == '__main__':
    main()
