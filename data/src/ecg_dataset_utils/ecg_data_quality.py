# --- Baseline packages ---
import pandas as pd
pd.set_option('display.max_columns', None)

# --- Utility packages ---
import os
import time
import datetime



def check_missing_values(data_folder, file_header=True, total_rows=5000):
    """
    Checks for missing, zero, and incomplete readings in ECG CSV files and provides summaries.
    Assumes the data is in the format where each column represents and ECG lead and each row
    is a reading at a point in time.

    Arguments:
        data_folder (str): The path to the folder containing the ECG readings stored in CSV files.
        file_header (bool, optional): Flag to indicate whether the csv files contain headers. (Default: True)
        total_rows (int, optional): Number of rows with data. (Default: 5000)
    
    Returns:
        pd.Series: A Series of filenames that were identified as having issues, sorted alphabetically.
    """
    print(f'\n[{datetime.datetime.now()}]   Checking the data quality of ecg readings in {data_folder}')
    time_start = time.time()

    null_check = []
    zero_check = []
    missing_check = []
    files_with_issues = set()
    lead_columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    ecg_file_list = [fileName for fileName in sorted(os.listdir(data_folder)) if fileName.endswith('.csv')]
    if not ecg_file_list:
        print(f"No CSV files found in {data_folder}")

    for filename in ecg_file_list:
        # Read ECG readings csv file.
        if file_header:
            ecg_df = pd.read_csv(os.path.join(data_folder, filename))
        else:
            ecg_df = pd.read_csv(os.path.join(data_folder, filename), header=None, names=lead_columns)
        ecg_base_name = filename.replace(".csv", "")

        # --- Perform Checks ---
        # > 1. Check for missing values.
        null_vals = ecg_df.isnull().sum()
        null_dict = null_vals.to_dict()
        null_dict['filename'] = ecg_base_name
        null_check.append(null_dict)
        if null_vals.any():
            files_with_issues.add(ecg_base_name)
        
        # > 2. Check for zero values.
        zero_vals = (ecg_df==0).sum()
        zero_dict = zero_vals.to_dict()
        zero_dict['filename'] = ecg_base_name
        zero_check.append(zero_dict)
        if zero_vals.any():
            files_with_issues.add(ecg_base_name)
        
        # > 3. Identify files with incomplete rows.
        # >> Note: For the Chapman-Shaoxing database, all files should have 5000 rows (10s readings at 500Hz).
        missing_rows = total_rows - len(ecg_df)
        missing_check.append({'filename': ecg_base_name, 'missingrows': missing_rows})
        if missing_rows > 0:
            files_with_issues.add(ecg_base_name)

    # Convert lists of dictionaries to DataFrames.
    null_check_df = pd.DataFrame(null_check)
    zero_check_raw_df = pd.DataFrame(zero_check)
    missing_check_df = pd.DataFrame(missing_check)

    # Identify files with issues.
    rows_with_null = null_check_df[(null_check_df.drop(columns='filename') != 0).any(axis=1)]
    print(f"There are {len(rows_with_null)} files with missing data. This represents {len(rows_with_null)*100/len(null_check_df):.2f}% of the dataset.")

    rows_with_zero = zero_check_raw_df[(zero_check_raw_df.drop(columns='filename') != 0).any(axis=1)]
    print(f"There are {len(rows_with_zero)} files with zero values in the data. This represents {len(rows_with_zero)*100/len(zero_check_raw_df):.2f}% of the dataset.")

    rows_incomplete = missing_check_df[(missing_check_df.drop(columns='filename') != 0).any(axis=1)]
    print(f"There are {len(rows_incomplete)} files with incomplete rows in the data. This represents {len(rows_incomplete)*100/len(missing_check_df):.2f}% of the dataset.")

    rows_no_data = zero_check_raw_df[(zero_check_raw_df.drop(columns='filename') == total_rows).any(axis=1)]
    print(f"There are {len(rows_no_data)} files with all zero values in any of the leads. This represents {len(rows_no_data)*100/len(zero_check_raw_df):.2f}% of the dataset.")

    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f'[{datetime.datetime.now()}]   Summary: There are {len(files_with_issues)} unique files with issues. Time taken: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s!\n')

    return pd.Series(sorted(list(files_with_issues)))



def remove_problem_files(original_dataframe, problem_file_list, id_col='FileName', label_col='Rhythm'):
    """
    Generates a new metadata DataFrame containing only ECG recordings with complete information. This
    function filters original_dataframe to exclude any recordings whose IDs are present in problem_file_list.
    It also prints out the distribution of labels before and after ID removal.

    Arguments:
        original_dataframe (pd.DataFrame): The original metadata DataFrame.
        problem_file_list (pd.Series): A pandas Series containing unique IDs of identified having issues and to be removed.
        id_col (str, optional): The name of column in original_dataframe that contains the unique ID of each recording. (Default: FileName)
        label_col (str, optional): The name of column in original_dataframe that contains the target labels. (Default: Rhythm)
    
    Returns:
        pd.DataFrame: A new metadata DataFrame with problematic recordings removed.
    """
    # Remove values.
    clean_dataframe = original_dataframe[~original_dataframe[id_col].isin(problem_file_list)].copy()
    
    # Sort and reset index.
    clean_dataframe = clean_dataframe.sort_values(id_col).reset_index()

    # Compare target distribution before and after.
    print("Original dataset distribution:\n", original_dataframe[label_col].value_counts(normalize=True))
    print("\n")
    print("Modified dataset distribution:\n", clean_dataframe[label_col].value_counts(normalize=True))
    print("\n")
    print("Difference in number of observations:\n", original_dataframe[label_col].value_counts() - clean_dataframe[label_col].value_counts())

    return clean_dataframe