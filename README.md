🇪🇹 Amharic Character Recognition using TensorFlow & PyTorch

A deep learning project that builds and compares CNN-based Amharic character classifiers using both TensorFlow and PyTorch frameworks.
This project performs a full pipeline from preprocessing to evaluation and provides a detailed comparative performance analysis.

🚀 Project Overview

Amharic is a morphologically rich language with 237 unique character classes, making character recognition a challenging multi-class classification task.

In this project:

📦 37,652 grayscale images were processed

🧠 CNN architectures were built in both TensorFlow and PyTorch

📊 Models were trained, validated, and evaluated

⚖️ Performance was compared using Accuracy and F1-scores

🔍 Class-wise performance was analyzed

📊 Dataset Information

Total Images: 37,652

Number of Classes: 237 Amharic characters

Image Size: 64 × 64

Color Format: Grayscale

Normalization: Pixel values scaled to [0, 1]

Data Split

🟢 Training: 70%

🟡 Validation: 15%

🔵 Test: 15%

🧠 Model Architecture

Both frameworks used a similar Convolutional Neural Network (CNN) structure:

Convolutional Layers

ReLU Activations

Max Pooling

Fully Connected Layers

Softmax Output (237 classes)

Total Trainable Parameters (TensorFlow Model): 1,655,149

📈 Model Performance Comparison
Metric	TensorFlow	PyTorch
Test Accuracy	73.96%	71.29%
Macro F1-score	0.67	0.65
Weighted F1-score	0.74	0.71
🏆 Winner: TensorFlow (Slightly Better Overall Performance)
🔍 Detailed Observations
✅ Overall Accuracy

TensorFlow achieved 73.96%

PyTorch achieved 71.29%

TensorFlow shows a small but consistent advantage.

✅ Macro F1-score

TensorFlow: 0.67

PyTorch: 0.65
TensorFlow performs slightly better across all classes, including minority ones.

✅ Weighted F1-score

TensorFlow: 0.74

PyTorch: 0.71
Indicates better performance considering class imbalance.

🎯 Class-wise Performance Insights

Some characters achieved very high F1-scores (0.95+)

Some characters performed poorly (F1-score close to 0.00)

Difficult characters include visually similar shapes

Performance variance suggests:

Data imbalance

Insufficient variation for certain characters

Intrinsic visual similarity challenges

⚠️ Overfitting Observed
TensorFlow

Validation accuracy peaked at 75.05%

Validation loss increased after epoch 5
➡ Suggests overfitting

PyTorch

Validation accuracy peaked at 72.66%

Slight decline after epoch 7
➡ Also shows overfitting trend

🛠️ Key Challenges

237-class multi-class classification problem

High inter-class similarity

Class imbalance

Font rendering issues in visualization (does not affect training)

💡 Future Improvements
1️⃣ Reduce Overfitting

Add Dropout layers

Apply L1/L2 regularization

Implement Early Stopping

Use stronger data augmentation

2️⃣ Improve Difficult Classes

Class-weighted loss functions

Oversampling minority classes

Focused augmentation per weak class

3️⃣ Advanced Architectures

Transfer Learning (ResNet, EfficientNet)

Deeper CNNs

Batch Normalization layers

4️⃣ Deployment Ideas

🔤 Web-based Amharic OCR system

📱 Mobile character recognition app

📝 Amharic handwriting recognition tool

🧾 Document digitization system

🏗️ Tech Stack

Python

TensorFlow

PyTorch

NumPy

Matplotlib

Scikit-learn

📌 Conclusion

Both TensorFlow and PyTorch successfully built strong CNN models for Amharic character recognition.

While both models achieved reasonable performance (~70%+ accuracy), TensorFlow demonstrated slightly better generalization across evaluation metrics.

However, the task remains challenging due to:

Large number of classes

Visual similarity between characters

Data imbalance

This project demonstrates a complete end-to-end deep learning workflow and framework comparison for a real-world multi-class classification problem.

👨‍💻 Author

Segni Nadew
Machine Learning Engineer | Data Scientist | Full-Stack Developer
