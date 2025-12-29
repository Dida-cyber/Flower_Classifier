# Flower Classification Pipeline

This notebook implements a complete flower image classification pipeline using transfer learning with the MobileNetV2 model from TensorFlow/Keras.

##  Overview

The pipeline enables you to:
- Organize a flower image dataset into training and test sets
- Create a classification model based on MobileNetV2 (pre-trained model)
- Train the model on your custom data
- Evaluate model performance with detailed metrics

##  Results

The model was trained on a dataset containing **2 flower classes** (daisy and rose):
- **1,548 total images** (1,238 for training, 310 for testing)
- **Final accuracy: 94.84%** on the test set
- **16 errors** out of 310 test images

##  Dependencies

The notebook requires the following libraries:
- `tensorflow`: Deep learning framework
- `keras`: High-level API for TensorFlow
- `pillow`: Image processing
- `scikit-learn`: Machine learning tools (train_test_split, metrics)
- `matplotlib`: Visualization
- `seaborn`: Advanced visualization (confusion matrices)

##  Dataset Structure

The notebook expects an initial data structure:
```
flowers/
├── daisy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── rose/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

After execution, a new structure is created:
```
flowers_dest/
├── train/
│   ├── daisy/
│   └── rose/
└── test/
    ├── daisy/
    └── rose/
```

## 🔧 Main Features

### 1. Dataset Creation (`creer_dataset`)

This function:
- Automatically splits images into training (80%) and test (20%) sets
- Creates the necessary folder structure
- Copies images to appropriate folders
- Displays detailed statistics about the dataset

**Parameters:**
- `flowers`: Path to the source folder containing classes
- `flowers_dest`: Destination path for the organized dataset
- `test_size`: Proportion of dataset to use for testing (default: 0.2)

### 2. Model Creation (`creer_modele`)

Creates a classification model with:
- **MobileNetV2 Backbone**: Pre-trained model on ImageNet (frozen weights)
- **GlobalAveragePooling2D**: Dimensionality reduction
- **Dense Layer (128 neurons)**: Learning specific patterns
- **Dropout (50%)**: Regularization to prevent overfitting
- **Softmax Output Layer**: Multi-class classification

**MobileNetV2 Advantages:**
- Lightweight and fast
- Ideal for devices with limited resources
- Excellent performance thanks to transfer learning

### 3. Model Training (`entrainer_modele`)

This function:
- Applies **data augmentation** on the training set:
  - Rotation (±20°)
  - Horizontal/vertical translation (20%)
  - Horizontal flip
  - Zoom (±20%)
- Normalizes pixels (0-1)
- Trains the model with Adam optimizer
- Saves the model to `mon_classificateur.h5`
- Visualizes the evolution of accuracy and loss

**Parameters:**
- `flowers_dest`: Path to the organized dataset
- `num_classes`: Number of classes to classify
- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 32)

### 4. Model Evaluation (`evaluer_modele`)

Provides comprehensive evaluation with:
- **Classification Report**: Precision, Recall, F1-score per class
- **Confusion Matrix**: Visualization of classification errors
- **Error Analysis**: Identification of the most confident errors
- **Global Accuracy**: Accuracy metric on the test set

##  Evaluation Metrics

The model achieves the following performance:

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| daisy  | 0.95      | 0.94   | 0.95     | 153     |
| rose   | 0.94      | 0.96   | 0.95     | 157     |
| **Average** | **0.95** | **0.95** | **0.95** | **310** |

##  Usage

1. **Prepare your data**: Organize your flower images in the `flowers/` folder with one subfolder per class

2. **Run cells in order**:
   - Install dependencies
   - Import libraries
   - Create dataset
   - Create and train model
   - Evaluate model

3. **Results**: The trained model is saved in `mon_classificateur.h5`

##  Important Points

- **Transfer Learning**: The model uses pre-trained MobileNetV2, allowing good performance with limited data
- **Data Augmentation**: Improves model generalization by creating variations of training images
- **Regularization**: Dropout (50%) and freezing base layers help prevent overfitting
- **Reproducibility**: `random_state=42` ensures that train/test split is reproducible

## 🔍 Error Analysis

The notebook identifies errors where the model was most confident:
- These errors may indicate ambiguous or mislabeled images
- Useful for improving the dataset or model

##  Technical Notes

- **Image formats**: JPG, JPEG, PNG
- **Image size**: Resized to 224x224 pixels
- **Normalization**: Pixels normalized between 0 and 1
- **Optimizer**: Adam with learning rate of 0.001
- **Loss function**: Categorical crossentropy

##  Key Concepts

- **Transfer Learning**: Reusing a pre-trained model for a new task
- **Data Augmentation**: Generating new images from existing ones
- **Fine-tuning**: Adapting a pre-trained model to a new problem
- **Overfitting**: When the model memorizes training data instead of learning general patterns
