# EIGait: Environment Independent Gait Recognition Using Wi-Fi CSI

## Introduction
Wi-Fi signal multipath propagation presents a challenge as the signals reflected by the human body not only carry distinctive human features but are also distorted by the surrounding environment. To address this issue, we introduce **EIGait**, an advanced environment-independent gait recognition system leveraging Wi-Fi Channel State Information (CSI). Unlike conventional Wi-Fi-based gait recognition systems, **EIGait** aims to achieve high recognition accuracy while ensuring robustness against environmental changes.

Specifically, **EIGait** extracts human gait features by fusing time-frequency features from spectrograms with temporal features related to phase changes, capturing more comprehensive and distinctive gait patterns. It then transforms these features from different environments into a uniform, environment-independent feature space, ensuring consistent system performance across different environments.

To demonstrate the effectiveness of **EIGait**, we implemented it on commercial Wi-Fi devices and evaluated its performance across four distinct indoor environments. Experimental results demonstrate that **EIGait** outperforms existing gait recognition systems, consistently delivering superior performance even in challenging settings.

Key innovations of EIGait include:
- **Time-Frequency & Temporal Fusion**: Extracts gait features by fusing spectrogram-based time-frequency features with phase-change-related temporal features.
- **Environment-Independent Feature Space**: Transforms features from diverse environments into a unified representation to ensure consistent performance across settings.
- **Real-World Applicability**: Implemented on commercial Wi-Fi devices and validated across four distinct indoor environments.


## Model Implementation
The **EIGait** model, along with the models used for comparative experiments, is implemented in the `model/` directory. 

## Experimental Data & Scenarios
We evaluated **EIGait** in four different indoor environments to assess its robustness and accuracy. The experimental setups are illustrated in the following figures:
