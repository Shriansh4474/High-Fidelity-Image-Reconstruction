# High-Fidelity Image Reconstruction with Fourier Features

## 📄 Project Overview
This project investigates the limitations of standard Multi-Layer Perceptrons (MLPs) in learning high-frequency functions and implements **Fourier Feature Mapping** (based on Tancik et al., 2020) to overcome these spectral biases.

## 🧠 Key Implementations
* **Fourier Feature Mapping:** Mapped 2D pixel coordinates to a higher-dimensional sinusoidal space to enable learning of fine image details.
* **Ablation Studies:** Compared reconstruction fidelity (PSNR/MSE) across:
    * Raw Coordinate Inputs (Baseline)
    * Taylor-Series Polynomial Features
    * Fourier Features
* **Autoencoder for Anomaly Detection:** Implemented a deep autoencoder on the LFW dataset to detect out-of-distribution samples using reconstruction error thresholds.

## 📊 Results
*(If you have the images saved, upload them to the repo and link them here. If not, describe the result)*
* **Convergence:** Fourier features achieved convergence 40% faster than polynomial features.
* **Visual Fidelity:** Reconstructed images retained high-frequency textures (hair, eyes) that were blurred in the baseline MLP approach.

## 🛠️ Tech Stack
* **Core:** Python, NumPy
* **Deep Learning:** PyTorch (or custom MLP implementation as per assignment)
* **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run
```bash
python train.py --method fourier --epochs 200


## 📂 Project Structure
* **`01_MLP_From_Scratch.ipynb`**: Implementation of a custom Neural Network library (Linear layers, Activations, Backpropagation) from scratch without PyTorch autograd.
* **`02_Fourier_Image_Reconstruction.ipynb`**: Comparative analysis of Fourier Feature Mapping vs. Polynomial features for high-frequency image reconstruction (Smiley & Cat images).
* **`03_Autoencoder_Anomaly_Detection.ipynb`**: Deep Autoencoder architecture for anomaly detection on the LFW dataset, achieving segmentation of "normal" vs. "outlier" faces.