import numpy as np
import functions as f
import dataset_functions as df
import feacture_calculator as fc



def main():

    #Cleans and filters by removing malformed rows and ensuring the first column has between 4 and 30 characters.
    #f.clean_and_filter_csv('data.csv', 'data_filtered.csv')

    # Load dataset as Dataframe and convert it into Numpy array
    #file_path = 'data_filtered.csv'
    #df_dataset = df.load_data(file_path)
    #np_dataset = df.convert_DataFrame_to_Numpy(df_dataset)



    #Count the occurrences for each strength level and Plot of the class distribution
    #esult = df.count_strengths(np_dataset)
    #df.plot_strength_distribution(result)


    # calculate the password features, both normalized and unnormalized.
    #features_norm = fc.calculate_password_features(np_dataset, norm=True)
    #features = fc.calculate_password_features(np_dataset, norm=False)

    # Column headers
    #header = "password,strength,Digit Count,Lowercase Count,Uppercase Count,Special Char Count,Unique Char Count,Total Length"

    # Salvataggio su CSV
    #np.savetxt("password_features.csv", features, delimiter=",", fmt="%s", header=header, comments="", encoding="utf-8")
    #np.savetxt("password_features_norm.csv", features_norm, delimiter=",", fmt="%s", header=header, comments="", encoding="utf-8")
    #print(f"Correctly saved in: {file_path}")

    # Save passwords to a file
    # output_file = 'passwords.txt'
    # f.save_to_file(passwords, output_file)

    print()

if __name__ == '__main__':
    main()
