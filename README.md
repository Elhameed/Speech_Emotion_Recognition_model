# Speech Emotion Recognition (SER) Project

## Overview

This project proposes the development of a machine learning model for Speech Emotion Recognition (SER) using audio data. The primary objective is to classify speech into various emotional categories, such as happiness, sadness, frustration, etc. by analyzing audio features. Existing speech recognition systems predominantly focus on textual data, often neglecting the emotional tones present in speech. This limitation can hinder their effectiveness in sensitive applications, such as online learning environments, healthcare, customer service, and therapy. By leveraging a Convolutional Neural Network (CNN) architecture, the project aims to:
1. Build a Deep Learning Model: Create an effective model utilizing CNN to identify and classify speech into various emotional categories.

2. Optimize Model Performance: Experiment with various optimization techniques (including Adam and RMSprop), regularization methods (L1 and L2 regularization, dropout), and hyperparameter tuning to improve accuracy and minimize loss.

3. Evaluate Model Effectiveness: Assess the model's performance using key metrics such as accuracy, precision, recall, and F1 score to ensure reliable classification capabilities.

4. Perform Error Analysis: Utilize confusion matrices to visualize and analyze classification errors, helping identify areas for improvement in model performance.

5. Save and Deploy Model: Ensure that the trained model can be easily saved and deployed for future use or further enhancements.

## Dataset

The project utilizes the RAVDESS dataset, which comprises 1440 speech audio-only files (16-bit, 48kHz .wav). This dataset features a range of emotions, including calm, happy, sad, angry, fearful, surprised, and disgust, produced by 24 professional actors (12 male and 12 female). The dataset is instrumental in training the SER model, facilitating a nuanced understanding of emotional expression in speech. For training the model, only four actors (Actor_01 to Actor_04) were selected. This decision was made to:

- Simplify the training process and reduce the complexity of the model.
- Focus on a smaller, more manageable subset of data while still providing a diverse range of emotional expressions.
- Allow for faster experimentation and iteration during model development.

Link to the RAVDESS dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download

## Key Findings and Discussion

### Model Comparison

This project developed two models for Speech Emotion Recognition (SER): a **vanilla model** and an **optimized model**. The purpose of this comparison was to evaluate the impact of optimizations on model performance.

#### Vanilla Model (No Optimization)

Evaluation Results:

Test Loss: 1.5526
Test Accuracy: 0.6667

### Optimized Model 

Evaluation Results:

Test Loss: 1.4849
Test Accuracy: 0.7778

### Discussion

The results from the evaluation indicate a significant improvement in performance metrics when transitioning from the vanilla model to the optimized model. The optimized model achieved an accuracy of **77.78%**, compared to **66.67%** for the vanilla model.

Key observations include:

- **Regularization:** The use of L2 regularization in the optimized model helped reduce overfitting, resulting in better generalization on the test dataset.
- **Dropout Layer:** Implementing a dropout layer significantly improved the model's robustness by preventing overfitting during training.
- **Overall Performance:** The optimized model performed better in both loss and accuracy metrics, demonstrating that the applied optimization techniques effectively enhanced the model's ability to classify emotional states from speech.

Further analysis of the confusion matrices reveals that the optimized model managed to classify more emotional states correctly compared to the vanilla model, which misclassified many instances, especially in the "Happy" and "Sad" categories.
