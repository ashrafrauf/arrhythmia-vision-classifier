# --- Utility packages ---
import os
import sys
import re
import argparse
import json



def find_errors_in_logs(experiment_folder, capture_length=5):
    """
    Parses SLURM error log files within a specified experiment directory to find common error indicators and saves the findings to a JSON file.

    Arguments:
        experiment_folder (str): The name of the experiment folder.
        capture_length (int, optional): Number of context lines to capture. (Default: 5)

    Returns:
        dict: A dictionary where keys are error filenames and values are lists of context lines around the detected error. Returns an empty dict if no errors are found or if the log directory doesn't exist.
    """
    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    slurm_err_logs_dir = os.path.join(project_dir, "experiments", experiment_folder, "slurm-logs", "err")

    # Define common error patterns using regular expressions (case-insensitive)
    error_patterns = [
        re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
        re.compile(r"Error", re.IGNORECASE),
        re.compile(r"Exception", re.IGNORECASE),
        re.compile(r"Segmentation fault", re.IGNORECASE),
        re.compile(r"Killed", re.IGNORECASE), # Often indicates Out Of Memory (OOM) or process termination
        re.compile(r"FileNotFoundError", re.IGNORECASE),
        re.compile(r"ModuleNotFoundError", re.IGNORECASE),
        re.compile(r"CUDA error", re.IGNORECASE),
        re.compile(r"RuntimeError", re.IGNORECASE),
        re.compile(r"SyntaxError", re.IGNORECASE),
        re.compile(r"permission denied", re.IGNORECASE),
        re.compile(r"disk quota exceeded", re.IGNORECASE),
        re.compile(r"Bus error", re.IGNORECASE),
    ]

    print(f"Searching for errors in log files under: {slurm_err_logs_dir}")
    print("-" * 50)

    
    # Instantiate dictionary to store errors: {filename: [context_lines]}
    error_dict = {}

    # Instantiate variable to count errors.
    error_found_count = 0

    # Get list of error files.
    error_file_list = sorted(os.listdir(slurm_err_logs_dir))

    # Check to confirm there are log files. Otherwise, returns empty dictionary.
    if len(error_file_list)==0:
        print(f"No .err files found in {slurm_err_logs_dir}")
        return {}
    
    # Iterate through each found error file.
    for error_file in error_file_list:
        error_file_path = os.path.join(slurm_err_logs_dir, error_file)
        has_error = False
        
        try:
            with open(error_file_path, 'r') as f:
                lines = f.readlines()
                # Check each line for any of the defined error patterns
                for i, line in enumerate(lines):
                    for pattern in error_patterns:
                        if pattern.search(line):
                            has_error = True
                            # Capture a few lines after for context.
                            end_idx = min(len(lines), i + capture_length)
                            error_dict[error_file] = [l.strip() for l in lines[i:end_idx]]
                            # Found a pattern in this line, no need to check other patterns for this line
                            break
                    if has_error:
                        # Found an error in this file, no need to check further lines.
                        break

            if has_error:
                error_found_count += 1
                print(f"\nERROR DETECTED in file: {os.path.basename(error_file_path)}")
                print("--- Context ---")
                for l in error_dict[error_file]:
                    print(f"    {l}")
                print("---------------\n")
        
        except Exception as e:
            sys.stderr.write(f"Could not read file {error_file_path}: {e}\n")
    
    print("-" * 50)
    if error_found_count > 0:
        print(f"Summary: Found errors in {error_found_count} out of {len(error_file_list)} error log files.")
    else:
        print(f"Summary: No common errors detected in any of the {len(error_file_list)} error log files.")
    
    # Returns the list of errors.
    return error_dict





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder", type=str, required=True, help="The name of the experiment folder (e.g., 250705_2105_resnet18_head_config1_grid).")
    parser.add_argument("--capture_length", type=int, default=5, help="How many lines of errors to capture for context.")
    args = parser.parse_args()

    # Identify scipt location and project directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Check for errors.
    error_dict = find_errors_in_logs(args.experiment_folder, args.capture_length)
    
    # Save to disk
    filepath = os.path.join(project_dir, "experiments", args.experiment_folder, "slurm_error_summary.json")
    with open(filepath, 'w') as f:
        json.dump(error_dict, f, indent=4)
    print(f"\nError summary saved to: {filepath}\n")