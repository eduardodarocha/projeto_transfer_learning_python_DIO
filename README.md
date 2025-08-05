# Cat vs. Dog Image Classification (Google Colab)

This project uses a deep learning model to classify images of cats and dogs. It is designed to be run in a Google Colaboratory environment, leveraging transfer learning with a pre-trained convolutional neural network (CNN) to achieve high accuracy.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup in Google Colab](#setup-in-google-colab)
- [Running the Notebook](#running-the-notebook)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

## About the Project

This project implements a binary image classifier to distinguish between cats and dogs. The model is built using transfer learning, which allows for high performance without requiring a massive dataset or extensive training time from scratch.

### Libraries Used

*   **pandas:** For data manipulation and analysis.
*   **numpy:** For numerical operations.
*   **os:** For interacting with the operating system.
*   **keras:** The high-level neural networks API.
*   **matplotlib.pyplot:** For creating static, animated, and interactive visualizations.
*   **keras.layers.Dense:** For creating densely-connected NN layers.
*   **keras.layers.GlobalAveragePooling2D:** For global average pooling operation for spatial data.
*   **keras.applications.MobileNet:** The MobileNet model for transfer learning.
*   **keras.preprocessing.image:** For image preprocessing.
*   **keras.applications.mobilenet.preprocess_input:** For preprocessing input for the MobileNet model.
*   **tensorflow.keras.preprocessing.image.ImageDataGenerator:** For generating batches of tensor image data with real-time data augmentation.
*   **keras.models.Model:** The `Model` class used to create a Keras model.
*   **keras.optimizers.Adam:** The Adam optimizer.
*   **keras.layers.Dropout:** For adding dropout regularization.

## Getting Started

To get this project running in Google Colab, follow these steps.

### Prerequisites

*   A Google Account.
*   The project files, including the main Jupyter Notebook (`.ipynb`) and the dataset directories.

### Setup in Google Colab

1.  **Upload the Notebook:**
    *   Go to [Google Colab](https://colab.research.google.com/).
    *   Click on `File -> Upload notebook` and select the project's `.ipynb` file (e.g., `cat_vs_dog_classifier.ipynb`).

2.  **Prepare the Dataset:**
    *   The model requires the `training_set` and `test_set` directories.
    *   The recommended way to handle this is to zip the `training_set` and `test_set` folders, upload the zip file to your Google Drive, and then mount your Google Drive in the Colab notebook.
    *   The notebook should contain code cells to mount Google Drive and unzip the dataset into the Colab environment.

3.  **Install Dependencies:**
    *   The notebook should include cells at the beginning to install any necessary Python packages using `!pip install ...`. Run these cells first to ensure all dependencies are met.

## Running the Notebook

Once the setup is complete, you can run the project by executing the notebook cells in order from top to bottom.

*   **Data Loading and Preprocessing:** These cells will load the images from the unzipped dataset directories and prepare them for training.
*   **Model Training:** This section will define the CNN architecture, load the pre-trained weights (transfer learning), and train the model on the `training_set`.
*   **Model Evaluation:** After training, the model's performance will be evaluated using the `test_set`.
*   **Single Prediction:** The final cells will allow you to test the model on new images from the `single_prediction` directory.

## Directory Structure

The project's data is organized as follows:

```
.
├── single_prediction/  # Images for single predictions
├── test_set/           # Test dataset
│   ├── cats/
│   └── dogs/
├── training_set/       # Training dataset
│   ├── cats/
│   └── dogs/
├── cat_vs_dog_classifier.ipynb # Main Colab notebook (example name)
└── README.md
```

*   **`single_prediction/`**: Contains images for making individual predictions with the trained model.
*   **`test_set/`**: The dataset for evaluating the model's performance.
*   **`training_set/`**: The dataset for training the model.

## Acknowledgments

*   This project uses the [Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle.
*   Powered by Google Colaboratory.