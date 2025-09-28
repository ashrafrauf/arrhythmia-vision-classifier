# Raw Data Folder

## Introduction
The dataset used for this study is the Chapman-Shaoxing database (Zheng, Zhang, et al., 2020). The data was collected jointly by Chapman University and Shaoxing People’s Hospital (Shaoxing Hospital Zhejiang University School of Medicine), and consist of more than 10,000 12-lead resting ECG recordings, each sampled at 500Hz and with a duration of 10 seconds.

<br/>

## Data Source
This folder contains the ECG recordings in csv file format and can be downloaded here: https://doi.org/10.6084/m9.figshare.c.4560497
- The `ECGData.zip` is downloaded and stored in this directory with the folder name "ecg-data-raw". It contains the raw ECG recordings.
- The `ECGDataDenoised.zip` is downloaded and stored in this directory with the folder name "ecg-data-denoised". It containes ECG recordings that have been denoised by the authors.

<br/>

Alternatively, the dataset is also available on PhysioNet here: https://doi.org/10.13026/92ks-sq55. However, there are two main differences:
- The files are stored in WFDB format.
- The total ECG recording available was increased to ~45k.

<br/>

## Folder Structure

```
 raw-data
 ┣ ecg-data-raw
 ┗ ecg-data-denoised
```

<br>

## References
Zheng, J., Zhang, J., Danioko, S., et al. (2020) “A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients,” Scientific Data, 7(1), p. 48. Available at: https://doi.org/10.1038/s41597-020-0386-x.