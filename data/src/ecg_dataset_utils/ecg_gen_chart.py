# --- Baseline packages ---
import numpy as np
import pandas as pd

# --- Visualisation packages ---
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# --- Utility packages ---
from math import ceil
import os
import time
from tqdm import tqdm



# A lightweight and modified version of the ecg_plot package implementation.
# GitHub repo: https://github.com/dy1901/ecg_plot
# Opted to reimplement to make it lightweight and make it more flexible to generate other layouts.
def ecg_plot(
        ecg_matrix,
        sample_rate = 500,
        chart_title = None,
        lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        lead_order = None,
        num_columns = 4,
        row_height = 6,
        short_strip_length = 2.5,
        rhythm_strip_label = None,
        show_grid = True,
        style = None,
        line_width = 0.5
):
    """
    Plots the ECG recordings into ECG charts using Matplotlib.
    
    Arguments:
        ecg_matrix (array-like): ECG recordings stored in m x n array, where m is number of leads and n is number of readings.
        sample_rate (int, optional): The sampling rate of the recordings in Hz, used to derive the time intervals between each reading. (Default: 500)
        chart_title (str, optional): Chart title for the ECG chart. (Default: None)
        lead_index (list, optional): A list containing the lead names for each of the m columns in ecg_matrix. (Default: 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
        lead_order (list, optional): A list containing indices to be used for lead arrangements. If none, leads will be visualised in order. (Default: None)
        num_columns (int, optional): Number of columns for the leads to be visualised in the ECG chart. (Default: 4)
        row_height (int, optional): The height of each row of ECG strip in terms of number of small grid squares. Each small square represents 0.1mV. (Default: 6)
        short_strip_length (float, optional): The length of each lead strip, in number of seconds. (Default: 2.5)
        rhythm_strip_label (str, optional): The lead name from lead_index to be used as rhythm strip. (Default: None)
        show_grid (bool, optional): Flag to show grid squares.
        style (str, optional): If 'bw', generates a black-and-white chart. Otherwise, generates a chart with red grid squares and blue lines.
        line_width (float, optional): The width of the ECG trace line. (Default: 0.5)
    
    Returns:
        None
    """
    if lead_order is None:
        lead_order = list(range(0,len(ecg_matrix)))

    # Derive time interval between samples, number of rows, and number of samples per strip.
    time_step = 1.0/sample_rate
    num_leads = len(ecg_matrix)
    num_rows = int(ceil(num_leads / num_columns))
    num_samples = int(sample_rate * short_strip_length)

    plot_width = num_columns * short_strip_length
    plot_height = num_rows * row_height / 5
    ecg_fig, ecg_axs = plt.subplots(figsize=(plot_width, plot_height))
    
    ecg_fig.subplots_adjust(
        hspace = 0,
        wspace = 0,
        left = 0,
        right = 1,
        bottom = 0,
        top = 1
    )
    ecg_fig.suptitle(chart_title)

    x_min = 0
    x_max = num_columns * short_strip_length

    if rhythm_strip_label is not None:
        y_min = row_height/4 - ((num_rows+1)/2)*row_height
    else:
        y_min = row_height/4 - (num_rows/2)*row_height
    y_max = row_height/4

    ecg_axs.set_ylim(y_min,y_max)
    ecg_axs.set_xlim(x_min,x_max)

    if style=='bw':
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0.7)
    
    if show_grid:
        # Set major ticks for grid lines.
        ecg_axs.set_xticks(np.arange(x_min,x_max,0.2))  # Major horizontal grid lines every 0.2 units
        ecg_axs.set_yticks(np.arange(y_min,y_max,0.5))  # Major vertical grid lines every 0.5 units

        # Enable minor ticks and set 5 minor ticks between major ticks.
        ecg_axs.minorticks_on()
        ecg_axs.xaxis.set_minor_locator(AutoMinorLocator(5))

        # Draw major and minor grid lines.
        ecg_axs.grid(which='major', linestyle='-', linewidth=0.5, color=color_major)
        ecg_axs.grid(which='minor', linestyle='-', linewidth=0.5, color=color_minor)

        # Remove x and y ticks and labels.
        ecg_axs.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ecg_axs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # Plots normal ECG strips.
    for c in range(num_columns):
        for i in range(num_rows):
            if (c * num_rows + i < num_leads):
                t_lead = lead_order[c * num_rows + i]
                y_offset = -(row_height/2) * i  # Offsets the y-axis for each strip, if not in the first row.
                x_offset = short_strip_length * c  # Offsets the x-axis for each strip, if not in the first column.
                sample_offset = int(sample_rate * x_offset)  # Extracts the relevant interval of samples.

                if c > 0:
                    ecg_axs.plot(
                        [x_offset, x_offset],
                        [ecg_matrix[t_lead][0] + y_offset - 0.3, ecg_matrix[t_lead][0] + y_offset + 0.3],
                        linewidth=line_width,
                        color=color_line
                    )
                
                ecg_axs.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9)
                ecg_axs.plot(
                    np.arange(0, num_samples*time_step, time_step) + x_offset,
                    ecg_matrix[t_lead][sample_offset:(num_samples + sample_offset)] + y_offset,
                    linewidth=line_width,
                    color=color_line
                )
    
    # Plots rhythm strip, if required.
    if rhythm_strip_label is not None:
        lead_dict = {'I':0, 'II':1, 'III':2, 'aVR':3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11}
        rhythm_samples = int(num_samples * num_columns)
        rhythm_strip = ecg_matrix[lead_dict[rhythm_strip_label]][:rhythm_samples]
        y_offset = -(row_height/2) * (num_rows)

        ecg_axs.text(0.07, y_offset-0.5, rhythm_strip_label, fontsize=9)
        ecg_axs.plot(
            np.arange(0, rhythm_samples*time_step, time_step),
            rhythm_strip + y_offset,
            linewidth=line_width,
            color=color_line
        )



def generate_standard_ecg_chart(
        raw_data_dir,
        file_name_list,
        output_dir,
        csv_header = None,
        chart_title = None,
        sampling_rate = 500,
        lead_order = None,
        num_columns = 4,
        short_strip_length = 2.5,
        rhythm_strip_lead = None,
        suffix = '_v0_std'
):
    """
    Wrapper function to automate the ECG chart generation from a list of raw data files specified by file_name_list. This function reads each specified CSV
    file from `raw_data_dir`, plots it, and saves the generated chart to `output_dir`. The values in each file is assumed to be in microVolts, hence will
    be divided by 1000 for the purposes of plotting. It also assumes ECG recordings stored in m x n array, where m is number of leads and n is number of readings.
    
    Arguments:
        raw_data_dir (str): The directory where the raw data is stored in CSV files.
        file_name_list (array-like): A list containing the filenames (without file extension) of the recordings in raw_data_dir to be visualised as ECG charts.
        output_dir (str): The directory where the generated charts will be stored in PNG format.
        csv_header (bool, optional): Flag to indicate whether the csv files contain headers. (Default: None)
        chart_title (str, optional): Chart title for the ECG chart. (Default: None)
        sampling_rate (int, optional): The sampling rate of the recordings in Hz, used to derive the time intervals between each reading. (Default: 500)
        lead_order (list, optional): A list containing indices to be used for lead arrangements. If none, leads will be visualised in order. (Default: None)
        num_columns (int, optional): Number of columns for the leads to be visualised in the ECG chart. (Default: 4)
        short_strip_length (float, optional): The length of each lead strip, in number of seconds. (Default: 2.5)
        rhythm_strip_lead (str, optional): The lead name to be used as rhythm strip. (Default: None)
        suffix (str, optional): Suffix to be added at the end of the filename to indicate the layout ID. (Default: "_v0_std")
    
    Returns:
        None: Each chart will be saved in output_dir with the filename format {ID}_{suffix}.png
    """
    # Start timer.
    timeStart = time.time()

    # Loop over the files.
    for i, file in tqdm(enumerate(file_name_list)):
        # Import the data.
        # > Raw data is in microVolts while ecg_plot assumes Volts. Hence, divide by 1000.
        # > To use ecg_plot, the data has to be in mxn matrix format, where m are leads and n are samples in the time domain.
        filePath = os.path.join(raw_data_dir, file + ".csv")
        rawDataDF = pd.read_csv(filePath, header=csv_header)
        ecgDataDF = rawDataDF.T / 1000
        ecgDataMat = ecgDataDF.values

        ecg_plot(
            ecgDataMat,
            sample_rate = sampling_rate,
            chart_title = chart_title,
            lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            lead_order = lead_order,
            num_columns = num_columns,
            row_height = 6,
            short_strip_length = short_strip_length,
            rhythm_strip_label = rhythm_strip_lead,
            show_grid = True,
            style = None,
            line_width = 0.5
        )
        
        # Save the plot to disk.
        plt.savefig(os.path.join(output_dir, file + suffix + ".png"), dpi = 300, bbox_inches='tight')

        plt.close()
    
    # Get elapsed time.
    timeElapsed = time.time() - timeStart

    print(f"Time taken: {timeElapsed // 60:.0f}m {timeElapsed % 60:.0f}s! | {i+1} charts generated. | Output saved at: {output_dir}")
    return None