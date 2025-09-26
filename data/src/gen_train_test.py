# --- Baseline packages ---
import pandas as pd
pd.set_option('display.max_columns', None)

# --- Statistical packages ---
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Utility packages ---
import os
import datetime
import joblib

# --- Helper functions ---
import ecg_dataset_utils





if __name__ == "__main__":
    """
    Orchestrator script to generate train-test data splits. Steps taken:
        1. Checks for data issues.
        2. Generates clean metadata file by removing recordings with data issues, creating merged-label classes, and encoding the labels.
        3. Creates 80% train / 20% test split using sklearn's train_test_split function. Split is stratified by granular original labels.
        4. Clean metadata, train metadata, and test metadata are saved in the metadata directory.
        5. Label encoders are also saved in the metadata directory.
    """

    # --------------------- #
    # Check for Data Issues #
    # --------------------- #
    print(f"\n[{datetime.datetime.now()}]   Checking for data issues.")

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Identify relevant folders.
    raw_data_dir = os.path.join(project_dir, "raw-data", "ecg-data-raw")
    denoised_data_dir = os.path.join(project_dir, "raw-data", "ecg-data-denoised")
    metadata_dir = os.path.join(project_dir, "metadata")

    # --- Check data quality ---
    raw_files_with_issues = ecg_dataset_utils.check_missing_values(raw_data_dir)
    denoised_files_with_issues = ecg_dataset_utils.check_missing_values(denoised_data_dir, file_header=False)


    # ----------------------- #
    # Generate Clean Metadata #
    # ----------------------- #
    print(f'\n[{datetime.datetime.now()}]   Cleaning metadata.')

    # Load original metadata.
    excel_filepath = os.path.join(project_dir, "metadata", "patient_diagnostics.xlsx")
    diagnostics_df = pd.read_excel(excel_filepath)

    # # Replace 'SA' rhythm with 'SI' (assumed typo). 
    # > 'SA' does not appear in the source paper nor subsequent referenced papers.
    diagnostics_df['Rhythm'] = diagnostics_df['Rhythm'].replace('SA', 'SI')

    # Remove filenames with issues.
    clean_df = ecg_dataset_utils.remove_problem_files(
        diagnostics_df,
        denoised_files_with_issues,
        id_col='FileName',
        label_col='Rhythm'
    )

    # Process rhythm labels. Create merged classes as per indicated in Zheng et al (2020).
    merged_rhythm_dict = {
        'AFIB': 'AFIB',
        'AF': 'AFIB',
        'SB': 'SB',
        'SR': 'SR',
        'SI': 'SR',
        'ST': 'GSVT',
        'SVT': 'GSVT',
        'AT': 'GSVT',
        'AVNRT': 'GSVT',
        'SAAWR': 'GSVT',
        'AVRT': 'GSVT'
    }
    clean_df['RhythmMerged'] = clean_df['Rhythm'].map(merged_rhythm_dict)

    # Rename columns to indicate hierarchy. Larger numbers mean more granular labels.
    new_rhythm_colnames = {'Rhythm': 'Rhythm_L2', 'RhythmMerged': 'Rhythm_L1'}
    clean_df = clean_df.rename(columns=new_rhythm_colnames)

    # Count number of beat conditions (exclude NONE).
    clean_df['BeatCount'] = clean_df['Beat'].replace('NONE', '').str.split().map(len)

    # Encode labels.
    rhythm_l1_encoder = LabelEncoder()
    rhythm_l2_encoder = LabelEncoder()
    clean_df['Rhythm_L1_Enc'] = rhythm_l1_encoder.fit_transform(clean_df['Rhythm_L1'])
    clean_df['Rhythm_L2_Enc'] = rhythm_l2_encoder.fit_transform(clean_df['Rhythm_L2'])


    # ----------------------- #
    # Create Train-Test Split #
    # ----------------------- #
    print(f'\n[{datetime.datetime.now()}]   Creating train-test split.')

    # Split into 80-20 train-test set.
    # > Note: Stratify based on original rhythm class to allow possibility of experimenting based on lower hierarchy rhythm.
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42, shuffle=True, stratify=clean_df.Rhythm_L1)

    # Reset row index.
    train_df = train_df.sort_values('FileName').reset_index(drop=True)
    test_df = test_df.sort_values('FileName').reset_index(drop=True)

    print("Number of train samples:", len(train_df))
    print("Number of test samples:", len(test_df))
    print("Total number of samples:", len(train_df) + len(test_df))


    # ------------------------------ #
    # Save to Folder: Train-Test DFs #
    # ------------------------------ #
    print(f'\n[{datetime.datetime.now()}]   Saving to: {metadata_dir}.')

    # Save clean metadata.
    clean_metadata_filepath = os.path.join(metadata_dir, "patient_diagnostics_clean.csv")
    clean_df.to_csv(clean_metadata_filepath, index=False)
    print(f'Saved clean metadata as: {os.path.basename(clean_metadata_filepath)}')

    # Save train and test dataframe.
    train_filepath = os.path.join(metadata_dir, "split_train.csv")
    train_df.to_csv(train_filepath, index=False)

    test_filepath = os.path.join(metadata_dir, "split_test.csv")
    test_df.to_csv(test_filepath, index=False)
    print(f'Saved train file as {os.path.basename(train_filepath)} and test file as {os.path.basename(test_filepath)}')

    # Save label encoders.
    encoder_filepath = os.path.join(metadata_dir, "rhythm_encoders_dict.joblib")
    label_encoder_dict = {
        'rhythm_l1_enc': rhythm_l1_encoder,
        'rhythm_l2_enc': rhythm_l2_encoder
    }
    joblib.dump(label_encoder_dict, encoder_filepath)
    print(f'Saved label encoders as {os.path.basename(encoder_filepath)}')

    print(f'\n[{datetime.datetime.now()}]   All files saved!')