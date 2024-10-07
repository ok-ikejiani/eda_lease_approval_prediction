from tabulate import tabulate

def analyze_missing_values(df):
    """
    Analyze and display missing values in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze for missing values.

    This function calculates the total count and percentage of missing values for each feature (column) in the DataFrame,
    and prints the results in a neatly formatted table using the `tabulate` module.
    """

    # Calculate the total number of missing values per feature
    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values per feature
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Create a list to store the results for each feature
    missing_data = []

    # Loop through the missing values and their corresponding percentages
    for feature, count in missing_values.items():
        percentage = missing_percentage[feature]
        # Append the feature, count of missing values, and percentage to the list
        missing_data.append([feature, count, f"{percentage:.2f}%"])

    # Define the headers for the table
    headers = ["Feature", "#Missing", "%Missing"]

    # Print the missing values analysis using `tabulate`
    print("\nMissing Values Analysis:")
    print(tabulate(missing_data, headers=headers, tablefmt="grid"))


def analyze_categorical_features(df):
    """
    Analyze and display categorical features in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze for categorical features.

    This function identifies and prints the categorical features (columns with object data type) in the DataFrame.
    """
    categorical_features = df.select_dtypes(include=['object']).columns
    print("\nCategorical Features Analysis:")
    print("Feature: {}".format(categorical_features))
    return categorical_features