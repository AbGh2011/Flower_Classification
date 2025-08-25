# Flower_Classification
This project is an AI-based image classification system using CNN to identify five flower types (Daisy, Rose, Dandelion, Sunflower, Tulip). It tackles class imbalance and enhances with data augmentation, ensures interpretability. A learning journey in machine learning.

# Overview
This project implements a deep learning-based flower classification system using Convolutional Neural Networks (CNN) with the VGG16 architecture, fine-tuned and enhanced with L2 Regularization. The system classifies images into five flower categories: Daisy, Dandelion, Rose, Sunflower, and Tulip. Built with TensorFlow/Keras, it leverages transfer learning, data augmentation, and regularization techniques to achieve robust performance on a real-world dataset.
Key features include:

- Transfer Learning: Utilizes pre-trained VGG16 from ImageNet for feature extraction.
- Fine-Tuning: Adjusts the last layers of VGG16 for better accuracy on the flower dataset.
- L2 Regularization: Prevents overfitting by penalizing large weights.
- Data Augmentation: Enhances model generalization with rotations and flips.
- Evaluation: Tracks accuracy, recall, and precision metrics.
- Visualization: Displays predictions with class labels on test images.

This project was a comprehensive learning journey in building, training, and optimizing a CNN model from scratch.

![Example of model's result](readme_images/result.png)

# Dataset
The dataset is sourced from the "Flowers Dataset" on Kaggle (by imsparsh), containing approximately 2746 images for training and validating the model which across five flower classes. After downloading and preprocessing, it includes:

- Classes: Daisy, Dandelion, Rose, Sunflower, Tulip.
- Train/Validation Split: Approximately 80/20 split from the original dataset.
- Test Set: 924 extra images  for final evaluation.

# Installation
- Clone the repository, create a virtual environment, and install the requirements.txt file.
- Connect to your kernel and start exploring the project.

# Preprocessing
The preprocessing pipeline loads images, resizes them to 224x224, normalizes pixel values (divide by 255), and splits the data into training and validation sets using ImageDataGenerator.

# Model Architecture
The model is based on VGG16 with transfer learning, fine-tuned, and regularized with L2.

- Base Model: VGG16 (pre-trained on ImageNet) with frozen initial layers.
- Custom Layers: Flatten, Dense (256 units with L2), Dropout (0.5), Dense (5 units with Softmax).
- Regularization: L2 with λ=0.01 to prevent overfitting.
- Total Parameters: ~15 million (trainable after fine-tuning: ~2 million).

# Training
The model is trained with categorical crossentropy loss, Adam optimizer, Early Stopping, Learning Rate Scheduling.

# Evaluation
Model performance is evaluated on the validation set:

- Validation Accuracy: ~85% (after Fine-Tuning and L2).
- Validation Loss: ~0.5.

# Model Download
- To download the basic model, click on the link:
[Basic Model](https://drive.google.com/file/d/1yWtFqzPiZVsdVItWYzPFPaf8lTeqPCLC/view?usp=sharing)

- To download the Fine-Tuned model, click on the link:
[Fine-Tuned Model](https://drive.google.com/file/d/1lPkw4SaZzg7bAk5uVBGlBgUKIEAEDp6I/view?usp=sharing)