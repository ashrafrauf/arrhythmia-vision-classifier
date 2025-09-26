# Metadata Folder

## Introduction
This folder contains the relevant metadata for the project. This includes:
- Metadata from the original study, available here: https://doi.org/10.6084/m9.figshare.c.4560497
    - `RhythmNames.xlsx` contains the acronym and full names of the rhythm-related arrhythmia conditions. Saved as 'rhythm_names.xlsx'.
    - `ConditionNames.xlsx`contains the acronym and full names of the beat-related arrhythmia conditions. Saved as 'condition_names.xlsx'.
    - `Diagnostics.xlsx` contains the list of patient IDs along with their associated labels, age, and extracted morphological attributes. Saved as 'patient_diagnostics.xlsx'.
    - `AttributesDictionary.xlsx` contains the description of each attribute available in the diagnostics file. Saved as 'attributes_dictionary.xlsx'.
- Cleaned version of the diagnostics file, where ECG recordings with data issues were removed. Saved as 'patient_diagnostics_clean.xlsx'.
- The rhythm encoded object used in the process of cleaning the diagnostic file. Contains encoder for both granular and merged labels and stored as a dictionary. Saved as 'rhythm_encoder_dict.joblib'.
- The train and test split of the clean diagnostic file, saved as 'split_train.csv' and 'split_test.csv'.


<br/>

## Folder Structure
```
metadata
 ┣ attributes_dictionary.xlsx
 ┣ condition_names.xlsx
 ┣ patient_diagnostics.xlsx
 ┣ patient_diagnostics_clean.csv
 ┣ rhythm_encoders_dict.joblib
 ┣ rhythm_names.xlsx
 ┣ split_test.csv
 ┗ split_train.csv
 ```