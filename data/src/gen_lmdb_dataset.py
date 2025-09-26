# --- Baseline packages ---
import pandas as pd

# --- Utility packages ---
import os
import datetime
import argparse

# --- Helper functions ---
import ecg_dataset_utils


def main_script(
        dataset_type,
        config_name,
        label_col = 'Rhythm_L1_Enc'
):
    """
    Orchestrator function to automate the generation of LMDB datasets.
    """
    # ------------- #
    # Preliminaries #
    # ------------- #
    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Identify relevant folders.
    data_processed_dir = os.path.join(project_dir, "processed-data")
    data_metadata_dir = os.path.join(project_dir, "metadata")
    main_output_dir = os.path.join(data_processed_dir, config_name, dataset_type)

    # Identify train or test set.
    if dataset_type in ['train', 'test']:
        ref_meta_filepath = os.path.join(data_metadata_dir, f"split_{dataset_type}.csv")
        chart_output_dir = os.path.join(main_output_dir, "png-files")

        print(f'\n[{datetime.datetime.now()}]   Start generating LMDB datasets.')
        img_metadata = ecg_dataset_utils.generate_lmdb_dataset(
            chart_output_dir,
            ref_meta_filepath,
            main_output_dir,
            label_col = label_col
        )
    
    # Combines both train and test sets into one LMDB.
    elif dataset_type == 'full':
        img_metadata = []
        for data_mode in ['train', 'test']:
            ref_meta_filepath = os.path.join(data_metadata_dir, f"split_{data_mode}.csv")
            chart_output_dir = os.path.join(data_processed_dir, config_name, data_mode, "png-files")

            print(f'\n[{datetime.datetime.now()}]   Start generating LMDB datasets for {data_mode} set samples.')
            mode_metadata = ecg_dataset_utils.generate_lmdb_dataset(
                chart_output_dir,
                ref_meta_filepath,
                main_output_dir,
                label_col = label_col
            )
            img_metadata = img_metadata + mode_metadata
    
    # Save dataframe.
    labels_csv_path = os.path.join(main_output_dir, "labels_keys.csv")
    img_metadata_df = pd.DataFrame(img_metadata)
    img_metadata_df.to_csv(labels_csv_path, index=False)
    print(f"\n[{datetime.datetime.now()}]   Done! Metadata CSV saved to {labels_csv_path}")

    print(f'\n[{datetime.datetime.now()}]   Done generating LMDB datasets!')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset args.
    parser.add_argument("--dataset_type", type=str, default='train', help="Specifies whether to process the 'train' or 'test' data split.")
    parser.add_argument("--config_name", type=str, default='new_config', help="A unique name for the dataset configuration")

    # LMDB dataset args.
    parser.add_argument("--label_col", type=str, default='Rhythm_L1_Enc', help="The name of the column in the metadata file that contains the labels for the images.")

    args = parser.parse_args()

    main_script(
        args.dataset_type,
        args.config_name,
        args.label_col
    )