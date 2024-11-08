# ANN-Thevenin-Hybrid-Model

![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

The **ANN-Thevenin-Hybrid-Model** is a hybrid machine learning model designed to predict the State of Charge (SoC) of a battery based on input features such as Voltage, Current, and Time. This model combines the predictive capabilities of Artificial Neural Networks (ANN) with the Thevenin equivalent model for enhanced accuracy and reliability.

## Features

- **Data Preprocessing**: Handles missing data and normalizes input features.
- **ANN Model**: Implements a deep neural network with multiple hidden layers.
- **Custom Metrics**: Utilizes SMAPE (Symmetric Mean Absolute Percentage Error) for evaluation.
- **Visualization**: Generates insightful plots to assess model performance.
- **Model Persistence**: Saves the trained model and scaler for future use.
- **Prediction Functionality**: Provides easy-to-use functions for making new predictions.

## Data

The model uses a dataset named `battery_text.csv`, which contains the following columns:

- **Voltage**: Voltage of the battery (in volts).
- **Current**: Current flow (in amperes).
- **Time**: Time duration (in seconds).
- **SoC**: State of Charge of the battery (target variable).

**Note**: Ensure that the dataset is placed in the `/content/` directory or update the path accordingly in the notebook.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ANN-Thevenin-Hybrid-Model.git
   cd ANN-Thevenin-Hybrid-Model
   ```
2. **Create a Virtual Environment** 
   It's recommended to use a virtual environment to manage dependencies.
   ```python
   python3 -m venv venv
   source venv/bin/activate
   ```
   Install Dependencies
   ```
   pip install -r requirements.txt
   ```
   Dependencies Include:
   ```
   python
   numpy
   pandas
   matplotlib
   seaborn
   scikit-learn
   tensorflow
   pickle-mixin
   ```

## Usage
1. Open the Notebook: Open ann_hybrid_github.ipynb using Jupyter Notebook or Google Colab.
2. Run Cells Sequentially Execute each cell in the notebook to perform data loading, preprocessing, model training, evaluation, and visualization.
3. Making Predictions At the end of the notebook, there is a section for making new predictions using the trained model. 
4. Update the voltage, current, and time variables with your desired input values and run the prediction cell.

## Model Architecture
The ANN model is built using TensorFlow's Keras API with the following structure:

1. Input Layer: Accepts three features - Voltage, Current, and Time.
2. Hidden Layers:
   1. Dense layer with 64 neurons and ReLU activation.
   2. Dense layer with 128 neurons and ReLU activation.
   3. Dense layer with 128 neurons and ReLU activation.
3. Output Layer: Single neuron for predicting SoC.
4. Compilation:

   1. Optimizer: Adam
   2. Loss Function: Mean Squared Error (MSE)
   3. Training Parameters:
   4. Epochs: 100
   5. Batch Size: 32
5. Evaluation:
After training, the model is evaluated using the following metrics:

   1. Mean Absolute Error (MAE)
   2. R² Score
   3. Symmetric Mean Absolute Percentage Error (SMAPE)

## Visualizations
The notebook generates several plots to visualize the model's performance:

1. Actual vs Predicted SoC (ANN): Scatter plot comparing actual SoC values against predictions.

2. Distribution of Errors (Histogram): Shows the distribution of prediction errors.

3. Residuals vs. Predicted Values: Helps identify any patterns in the residuals.

4. Correlation Matrix: Displays correlations between Voltage, Current, Time, and SoC.

5. Neural Network Diagram: Visual representation of the ANN architecture.
These visualizations aid in diagnosing model performance and understanding relationships within the data.

## Saving and Prediction
Saving
1. Scaler: The StandardScaler object is saved as ``scaler_X.pkl`` using pickle for consistent data preprocessing during inference.
2. Model: The trained ANN model is saved as ``hybrid_ann_thevenin_model.keras``.

3. Functions are provided to load the scaler and model and make predictions with new input data.
   
## License
This project is licensed under the MIT License.
