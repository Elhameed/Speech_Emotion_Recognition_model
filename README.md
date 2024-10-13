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

![vanilla Plot](https://github.com/Elhameed/zero_day/blob/master/Screenshot%20from%202024-10-13%2021-51-39.png)

### Optimized Model 

Evaluation Results:

Test Loss: 1.4849

Test Accuracy: 0.7778

![Optimized Plot](https://github.com/Elhameed/zero_day/blob/master/Screenshot%20from%202024-10-13%2021-52-02.png)

### Discussion

The results from the evaluation indicate a significant improvement in performance metrics when transitioning from the vanilla model to the optimized model. The optimized model achieved an accuracy of **77.78%**, compared to **66.67%** for the vanilla model.

### Key Observations

The transition from the vanilla model to the optimized model demonstrated notable improvements in performance metrics, particularly in terms of accuracy and loss. The optimized model achieved an accuracy of 77.78%, while the vanilla model recorded 66.67%. This enhancement can be attributed to several optimization techniques employed:

### 1. Regularization
- **Technique:** L2 Regularization
- **Principle:** L2 regularization adds a penalty for larger weights in the loss function, which helps prevent overfitting by discouraging complex models that fit the training data too closely.
- **Relevance:** By incorporating L2 regularization in both convolutional and dense layers, the optimized model generalized better on unseen data, leading to improved performance on the test set.
- **Parameter Significance:** The L2 penalty term was set to 0.01, chosen based on experimentation beacuse this performs best in accuracy. 
### 2. Dropout Layer
- **Technique:** Dropout
- **Principle:** Dropout randomly sets a fraction of the input units to 0 during training, which forces the network to learn robust features that are less reliant on specific neurons.
- **Relevance:** Implementing a dropout layer with a rate of 0.5 significantly reduced the chances of overfitting, enhancing the model's robustness and performance during evaluation.
- **Parameter Significance:** The 0.5 dropout rate was chosen based on standard practices in deep learning, aiming to strike a balance between retaining enough information and promoting generalization.
### 3. Adam Optimization
- **Technique:** Adam Optimizer
- **Principle:** Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. It computes adaptive learning rates for each parameter by using first and second moments of the gradients.
- **Relevance:** By utilizing Adam, the optimized model benefited from faster convergence and improved performance. This optimizer is particularly effective for problems with sparse gradients, making it suitable for audio data classification.
### 4. Model Architecture
- **Technique:** Sequential Model Structure
- **Principle:** A well-structured model architecture enhances learning by allowing deeper feature extraction through multiple convolutional layers followed by pooling layers.
- **Relevance:** The addition of multiple convolutional layers (with increasing filter sizes) enabled the model to capture intricate patterns in the audio data, facilitating better emotion classification.
  
### Overall Performance
- The optimized model demonstrated superior performance across various metrics, confirming that the integration of these optimization techniques substantially improved its ability to classify emotional states from speech.
- **Confusion Matrix Analysis:** Further examination of the confusion matrices highlighted that the optimized model correctly classified a wider range of emotional states compared to the vanilla model. The vanilla model struggled particularly with the "Happy" and "Sad" categories, indicating that the applied optimizations were essential for improving the model's overall accuracy and reliability. Here’s the data presented in a tabular format:

| Model                         | Accuracy  | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
|-------------------------------|-----------|-------------------|----------------|-------------------|
| Vanilla Model                 | 50%    | 52.50%            | 45.09%         | 45.23%            |
| Model with Optimization Technique | 66.67% | 62.98%            | 58.78%         | 59.48%            |

These optimization techniques, along with careful parameter selection and tuning, significantly contributed to the enhanced performance of the optimized model in recognizing emotions from speech data.

## Instructions for Running the Notebook

### Load the Notebook:
1. Open the Jupyter Notebook file (`your_notebook.ipynb`) in your preferred environment.

### Load the Dataset:
1. Ensure the RAVDESS dataset is organized in the specified directory structure.
2. Load the dataset into the notebook using the appropriate data loading functions.

### Run the Cells:
1. Execute the cells sequentially to preprocess the data, train the models, and evaluate their performance.
2. Monitor the output for training progress and evaluation metrics.

To load the trained models, you can use the following code snippets in your notebook:

### Loading the Vanilla Model
```python
from tensorflow.keras.models import load_model

vanilla_model = load_model('vanilla_model.h5')

### Loading the Optimized Model
```python
from tensorflow.keras.models import load_model

vanilla_model = load_model('optimized_model.h5')

```

## Conclusion
This README provides an overview of the Speech Emotion Recognition project, the dataset used, key findings, and instructions for running the notebook and loading models. By implementing these techniques and following the guidelines, you can effectively utilize the trained models for various applications in understanding emotional states through speech.

