# medical-image-Classification-TensorFlow
Medical image classification using CNN-based transfer learning in TensorFlow to detect clinically relevant classes from medical imaging data.

📌 **Overview**

This project implements a deep learning model to classify chest X-ray images as either NORMAL or PNEUMONIA using transfer learning with TensorFlow/Keras.

The goal is to demonstrate:

- End-to-end medical image classification

- Proper evaluation beyond accuracy (recall, ROC-AUC, confusion matrix)

- Threshold tuning for clinically meaningful trade-offs

🧠 **Problem Statement**

Chest X-rays are commonly used to diagnose pneumonia,
but manual interpretation can be time-consuming and error-prone.

- This project trains a convolutional neural network to:

- Detect pneumonia cases with high sensitivity

- Maintain reasonable specificity for normal cases

- Provide confidence scores rather than just hard labels

📂 **Dataset**

Source: Kaggle – Chest X-Ray Images (Pneumonia)

link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

Classes:

- NORMAL

- PNEUMONIA

Data Split:

- Training set

- Validation set

Test set (held out for final evaluation)

Note: The validation set is intentionally small, which can cause metric fluctuations. Final conclusions are based on the test set.

🏗️ **Model Architecture**

Backbone: Pretrained CNN (Transfer Learning)

- Custom Classification Head

- Dropout for regularization

- Sigmoid output for binary classification

Loss Function:

- Binary Cross-Entropy

Optimizer:

- Adam (learning rate = 1e-4)

⚙️ Training Details

Image size: 224 × 224

- Batch processing with tf.data

- Trained for multiple epochs with validation monitoring

- Data augmentation applied to reduce overfitting

📊 **Evaluation Metrics**

Instead of relying on accuracy alone, the model is evaluated using:

- Confusion Matrix

- Precision / Recall / F1-Score

- ROC Curve

- ROC-AUC

- Threshold analysis

✅ *Test Set Performance*
Metric	NORMAL	PNEUMONIA
Precision	0.85	0.92
Recall	0.87	0.91
F1-Score	0.86	0.91

- Overall Accuracy: 0.89

- ROC-AUC: 0.959

- An AUC of 0.959 indicates excellent separability between normal and pneumonia cases.

📈 **ROC Curve Interpretation**

The ROC curve illustrates the trade-off between:

- True Positive Rate (Sensitivity)

- False Positive Rate (1 − Specificity)

- A steep curve near the top-left corner shows that the model:

- Detects pneumonia with high sensitivity

- Keeps false alarms relatively low

This is especially important in medical screening applications.

🎯 **Threshold Optimization**

Rather than using the default 0.5 cutoff, multiple thresholds were evaluated to balance:

- Pneumonia recall (avoiding missed cases)

- Normal recall (avoiding unnecessary alarms)

- A threshold of ~ 0.8 was selected to achieve balanced clinical performance.

🖼️ **Single Image Inference**

The project includes a helper function to:

- Load a single X-ray image

- Output the predicted class

- Display the confidence score

Example:

Pred: NORMAL | Confidence: 0.84
Pred: PNEUMONIA | Confidence: 0.99

🧪 **Key Takeaways**

Transfer learning is highly effective for medical imaging tasks,
ROC-AUC provides a more reliable evaluation than accuracy alone.
Threshold tuning is essential for real-world medical deployment, 
and Confidence scores help interpret model predictions responsibly.

🚧 **Limitations & Future Work**

Expand validation set size,Test on external datasets.
Add Grad-CAM for visual explainability,
compare with Precision-Recall curves for class imbalance,
and explore multi-class pneumonia subtypes.

🛠️ **Technologies Used**

- Python

- TensorFlow / Keras

- NumPy

- Scikit-learn

- Matplotlib

- Jupyter Notebook

📎 *Disclaimer*

This project was for educational purposes only and wasn't intended for clinical diagnosis.
