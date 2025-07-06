# TML25 Assignment 3 – Robust Classifier under Adversarial Attacks

📌 **Additional Notes**
This repository contains all code and supporting files for training, adversarially fine-tuning, and exporting a robust image classifier as required for Assignment 3 of Trustworthy Machine Learning.

---

## Overview

This project aims to develop a robust image classification model that achieves high accuracy on both clean and adversarially perturbed data. The adversarial attacks considered are Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). The final submission is evaluated on a private test set for clean accuracy, as well as accuracy under both adversarial attacks, using a dedicated submission server.

---

## 👥 Team #17

* Maitri Shah - 7075780
* Yashashri Balwaik - 7075733

---

## Files

* `assignment3_solution.py`
  → Python script implementing the full pipeline:

  * Data loading, preprocessing, and augmentation
  * Model definition (ResNet-18) and training on clean data
  * Adversarial training using on-the-fly PGD attacks (with mix-clean strategy)
  * Validation and checkpointing
  * Saving final model state dict for submission

* `final_submission_model01.pt`
  → Final trained model weights (state\_dict) for submission.

* `README.md`
  → Summary of the method, key files, and instructions to reproduce results.

* `TML_A3_17_Report.pdf`
  → Detailed report explaining our solution, including design decisions, training details, and results.

---

## Approach

### Data Preparation & Augmentation

* All images are loaded and converted to 3x32x32 format as required by ResNet architectures.
* Data augmentation includes random cropping, horizontal flipping, rotation, color jitter, and random erasing to improve model generalization and reduce overfitting.

### Model Architecture & Training

* The classifier uses the ResNet18 backbone from torchvision, with its final layer adapted for 10-class output.
* Training is conducted in two phases:

  1. **Clean Training:** 40 epochs of supervised learning on clean images to achieve strong baseline accuracy.
  2. **Adversarial Training:** 15+ epochs of adversarial fine-tuning using PGD-generated examples. Each batch's loss is averaged between clean and adversarial samples (mix-clean strategy) to preserve clean accuracy while improving robustness.

### Adversarial Example Generation

* PGD adversarial examples are generated on-the-fly during adversarial training, using ε = 4/255, α = 1/255, and 7 steps.
* The validation loop evaluates both clean accuracy and adversarial accuracy.

### Model Export & Submission

* Only the PyTorch model state\_dict is exported and submitted, following assignment requirements.
* The model is submitted to the evaluation server with the correct headers for model-name and token.

---

## Results

* **Clean Accuracy (validation server):** \~57.5%
* **FGSM Accuracy:** \~29.6%
* **PGD Accuracy:** \~10.4%

The model exceeds the minimum clean accuracy requirement and demonstrates some robustness to adversarial attacks, with further improvements possible via more extensive adversarial training and hyperparameter tuning.

---

## Future Work

* Increase the duration and strength of adversarial training (e.g., more PGD steps or higher ε).
* Incorporate FGSM adversarial examples directly into training batches for broader robustness.
* Experiment with larger ResNet variants or alternative optimizers (e.g., SGD with momentum).
* Test label smoothing and advanced regularization techniques for improved robustness.

---

## How to Run

1. Ensure all dependencies are installed (`torch`, `torchvision`, `numpy`, `requests`, etc.).
2. Place the provided training dataset (`Train.pt`) in the working directory.
3. Run `assignment3_solution.py` to execute the pipeline: data loading, training, adversarial training, and model saving.
4. The final model weights will be saved as `final_submission_model01.pt`.
5. Submit this file to the evaluation server as per the assignment instructions.
