# --- Baseline packages ---
import pandas as pd

# --- Utility packages ---
import os
import datetime
import argparse

# --- Helper functions ---
import ecg_dataset_utils



def generate_image_files(
        dataset_type = 'train',
        config_name = 'new_config',
        sampling_rate = 500,
        short_strip_length = 2.5,
        lead_order = None,
        num_columns = 4,
        rhythm_strip_lead = 'II',
        image_suffix = '_v1_std'
):
    """
    Orchestrator function to automate the generation of ECG charts from ECG recording data.

    Arguments:
        dataset_type (str, optional): Specifies whether to process the 'train' or 'test' data split. (Default: 'train')
        config_name (str, optional): A unique name for this specific chart configuration. This name will be used to
                                           create a subdirectory within 'processed-data' to store the outputs. (Default: 'new_config')
        sampling_rate (int, optional): The sampling rate of the recordings in Hz, used to derive the time intervals between each reading. (Default: 500)
        short_strip_length (float, optional): The length of each lead strip, in number of seconds. (Default: 2.5)
        lead_order (list, optional): A list containing indices to be used for lead arrangements. If none, leads will be visualised in order. (Default: None)
        num_columns (int, optional): Number of columns for the leads to be visualised in the ECG chart. (Default: 4)
        rhythm_strip_lead (str, optional): The lead name to be used as rhythm strip. (Default: II)
        image_suffix (str, optional): Suffix to be added at the end of the filename to indicate the layout ID. (Default: "_v1_std")
    """
    # ------------- #
    # Preliminaries #
    # ------------- #
    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Identify relevant folders.
    data_raw_dir = os.path.join(project_dir, "raw-data", "ecg-data-denoised")
    data_processed_dir = os.path.join(project_dir, "processed-data")
    data_metadata_dir = os.path.join(project_dir, "metadata")

    # Identify train or test set.
    if dataset_type=='train':
        ref_meta_filepath = os.path.join(data_metadata_dir, "split_train.csv")
    elif dataset_type=='test':
        ref_meta_filepath = os.path.join(data_metadata_dir, "split_test.csv")
    else:
        raise ValueError("Please choose either 'train' or 'test' for dataset type.")
    print(f'\n[{datetime.datetime.now()}]   Creating {dataset_type} images for {config_name}.')

    # Get relevant metadata.
    ref_meta_df = pd.read_csv(ref_meta_filepath)
    filenames_list = list(ref_meta_df.FileName.values)
    
    # Create output directory.
    main_output_dir = os.path.join(data_processed_dir, config_name, dataset_type)
    chart_output_dir = os.path.join(main_output_dir, "png-files")
    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(chart_output_dir, exist_ok=True)
    print(f'    Saving generated charts to: {chart_output_dir}')


    # ------------------- #
    # Generate ECG Charts #
    # ------------------- #
    print(f'\n[{datetime.datetime.now()}]   Start generating ECG charts.')
    ecg_dataset_utils.generate_standard_ecg_chart(
        data_raw_dir,
        filenames_list,
        chart_output_dir,
        chart_title = None,
        sampling_rate = sampling_rate,
        lead_order = lead_order,
        num_columns = num_columns,
        short_strip_length = short_strip_length,
        rhythm_strip_lead = rhythm_strip_lead,
        suffix = image_suffix
    )
    print(f'\n[{datetime.datetime.now()}]   Done generating ECG charts!')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset args.
    parser.add_argument("--dataset_type", type=str, default='train', help="Specifies whether to process the 'train' or 'test' data split.")
    parser.add_argument("--config_name", type=str, default='new_config', help="A unique name for the dataset configuration")

    # ECG chart args.
    parser.add_argument("--sampling_rate", type=int, default=500, help="The sampling rate of the recordings in Hz, used to derive the time intervals between each reading.")
    parser.add_argument("--short_strip_length", type=float, default=2.5, help="The length of each lead strip, in number of seconds.")
    parser.add_argument("--num_columns", type=int, default=4, help="Number of columns for the leads to be visualised in the ECG chart.")
    parser.add_argument("--rhythm_strip_lead", type=str, default='II', help="The lead name to be used as rhythm strip. Parse 'None' to omit rhythm strip.")
    parser.add_argument("--image_suffix", type=str, default='_v1_std', help="Has to be in the format '_v{int}_std' for standard layout or '_v{int}_alt' for alternate layout. Int represents layout ID.")

    args = parser.parse_args()

    if args.rhythm_strip_lead=="None":
        rhythm_strip = None
    else:
        rhythm_strip = args.rhythm_strip_lead
    
    # Std refers to standard layout. Alt refers to alternate layout, similar to the one used in Sangha et al (2022).
    if args.image_suffix[-3:] == 'std':
        lead_order = None
    elif args.image_suffix[-3:] == 'alt':
        lead_order = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    else:
        raise ValueError('Provide suffix that ends with std for standard or alt for alternate image representations.')
    
    generate_image_files(
        dataset_type = args.dataset_type,
        config_name = args.config_name,
        sampling_rate = args.sampling_rate,
        short_strip_length = args.short_strip_length,
        lead_order = lead_order,
        num_columns = args.num_columns,
        rhythm_strip_lead = rhythm_strip,
        image_suffix = args.image_suffix
    )