🇪🇹 Amharic Character Recognition using TensorFlow & PyTorch
<p align="center"> <b>Deep Learning • Computer Vision • Multi-Class Classification • Framework Comparison</b> </p> <p align="center"> A complete end-to-end deep learning project that builds and compares CNN-based Amharic character classifiers using <b>TensorFlow</b> and <b>PyTorch</b>. </p>
📌 Project Overview

Amharic is a morphologically rich language containing 237 unique character classes, making character recognition a challenging large-scale multi-class classification problem.

This project implements the full deep learning pipeline:

📦 Processed 37,652 grayscale images

🧠 Built CNN architectures in TensorFlow and PyTorch

📊 Trained, validated, and tested both models

⚖️ Compared performance using Accuracy & F1-score

🔍 Conducted class-wise performance analysis

📊 Dataset Information
Property	Value
Total Images	37,652
Number of Classes	237
Image Size	64 × 64
Color Format	Grayscale
Normalization	Pixel values scaled to [0, 1]
📂 Data Split

🟢 Training: 70%

🟡 Validation: 15%

🔵 Test: 15%

🧠 Model Architecture

Both frameworks use a similar Convolutional Neural Network (CNN) structure:

Convolutional Layers

ReLU Activations

Max Pooling

Fully Connected Layers

Softmax Output Layer (237 classes)

TensorFlow Model Parameters:
1,655,149 trainable parameters

📈 Model Performance Comparison
Metric	TensorFlow	PyTorch
Test Accuracy	73.96%	71.29%
Macro F1-score	0.67	0.65
Weighted F1-score	0.74	0.71
🏆 Overall Winner: TensorFlow

TensorFlow achieved slightly better generalization across all evaluation metrics.

🔍 Detailed Observations
✅ Accuracy

TensorFlow: 73.96%

PyTorch: 71.29%

TensorFlow shows a small but consistent advantage.

✅ Macro F1-score

TensorFlow performs slightly better across minority classes.

✅ Weighted F1-score

Indicates better handling of class imbalance.

🎯 Class-wise Performance Insights

Some characters achieved F1-scores above 0.95

Some classes scored near 0.00

Difficult classes include visually similar characters

Performance variance suggests:

Data imbalance

Limited variation in some classes

Intrinsic visual similarity challenges

⚠️ Overfitting Observed
TensorFlow

Validation accuracy peaked at 75.05%

Validation loss increased after epoch 5
➡ Indicates overfitting

PyTorch

Validation accuracy peaked at 72.66%

Slight decline after epoch 7
➡ Also shows overfitting trend

🛠 Key Challenges

237-class multi-class classification problem

High inter-class similarity

Class imbalance

Visualization font rendering limitations

💡 Future Improvements
1️⃣ Reduce Overfitting

Dropout layers

L1/L2 regularization

Early Stopping

Stronger data augmentation

2️⃣ Improve Difficult Classes

Class-weighted loss functions

Oversampling minority classes

Focused augmentation

3️⃣ Advanced Architectures

Transfer Learning (ResNet, EfficientNet)

Deeper CNNs

Batch Normalization

4️⃣ Deployment Ideas

🔤 Web-based Amharic OCR system

📱 Mobile recognition app

📝 Handwriting recognition tool

🧾 Amharic document digitization system

🏗️ Tech Stack

Python

TensorFlow

PyTorch

NumPy

Matplotlib

Scikit-learn

🎯 Project Highlights

✔ Cross-framework comparison (TensorFlow vs PyTorch)
✔ Large-scale multi-class classification (237 classes)
✔ Evaluation beyond accuracy (Macro & Weighted F1-score)
✔ Real-world language-focused AI application

👨‍💻 Author

Segni Nadew
Machine Learning Engineer | Data Scientist | Full-Stack Developer
