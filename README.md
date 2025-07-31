# Handwritten Digit Classification using Logistic Regression

This project demonstrates the process of classifying handwritten digits (0-9) using a Logistic Regression model. It utilizes the `load_digits` dataset from scikit-learn, which is a classic dataset for introductory machine learning tasks.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Methodology](#methodology)
5.  [Results](#results)
6.  [Visualizations](#visualizations)
7.  [How to Run](#how-to-run)
8.  [Libraries Used](#libraries-used)
9.  [Acknowledgements](#acknowledgements)

## 1. Project Overview

The primary goal of this project is to implement and evaluate a Logistic Regression model for multi-class classification. We aim to correctly identify handwritten digits from a dataset of 8x8 pixel images. The project covers essential machine learning steps from data loading and exploration to model training and performance evaluation.

## 2. Dataset

The project uses the `load_digits` dataset available in `sklearn.datasets`.
* **Source**: Scikit-learn's `load_digits` dataset.
* **Description**: This dataset consists of 1,797 8x8 pixel grayscale images of handwritten digits (0 through 9).
* **Features (X)**: Each image is represented by a 64-dimensional array (8x8 pixels), where each value represents the pixel intensity. Pixel intensity values range from 0 to 16.
* **Target (y)**: The corresponding digit (0-9) that each image represents.

## 3. Exploratory Data Analysis (EDA)

EDA was performed to understand the structure and characteristics of the `load_digits` dataset.

* **Sample Digit Visualization**: Displays a grid of sample 8x8 images from the dataset, along with their true labels, to provide a visual understanding of the data.
* **Class Distribution**: A count plot shows the distribution of each digit (0-9) in the dataset, indicating if the dataset is balanced across classes. The `load_digits` dataset is generally well-balanced.
* **Pixel Intensity Distribution**: A histogram of all pixel intensity values (0-16) across the entire dataset helps understand the common intensity levels.
* **Correlation Heatmap**: A heatmap of pixel intensities indicates the correlation between different pixels. While less directly interpretable for image data, it can reveal patterns of co-occurrence.
* **2D PCA Visualization**: Principal Component Analysis (PCA) was applied to reduce the 64-dimensional data into 2 principal components. A scatter plot of these components, colored by digit class, helps visualize the separability of the different digit clusters in a lower-dimensional space.

## 4. Methodology

The following steps were performed to build and evaluate the Logistic Regression model:

1.  **Load Libraries**: All necessary libraries including `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.datasets`, `sklearn.model_selection`, `sklearn.preprocessing`, `sklearn.decomposition`, `sklearn.linear_model`, and `sklearn.metrics` were imported.
2.  **Load Dataset**: The `load_digits` dataset was loaded, separating features (`X`) from the target variable (`y`).
3.  **Data Splitting**: The dataset was split into training and testing sets using `train_test_split`, with 80% for training and 20% for testing. No `stratify` parameter was used in the provided notebook, which is a minor deviation from best practice for classification datasets to ensure class balance in splits, but for `load_digits` it often isn't critical due to its inherent balance.
4.  **Data Scaling**: `StandardScaler` was applied to the training and testing feature sets. This is a crucial preprocessing step for Logistic Regression, as it helps in faster convergence and better performance by standardizing feature scales. Note: The provided notebook applies `fit_transform` to `X_train` and `fit_transform` to `X_test`. For correct scaling, `transform` should be used on `X_test` after `fit_transform` on `X_train` to avoid data leakage. *This discrepancy is noted for potential improvement.*
5.  **Model Application (Logistic Regression)**: A `LogisticRegression` model was initialized with `max_iter=5000`, `random_state=42`, `solver='saga'`, and `multi_class='multinomial'`. The `max_iter` was increased to ensure convergence, and `saga` with `multinomial` is suitable for multi-class problems. The model was then trained using the scaled training data.
6.  **Prediction**: The trained model made predictions (`y_pred_lr`) on the scaled test set features.
7.  **Model Evaluation**: The performance of the Logistic Regression model was assessed using:
    * **Accuracy Score**: The proportion of correctly classified instances.
    * **Classification Report**: Provides precision, recall, and f1-score for each class, as well as macro and weighted averages.
    * **Confusion Matrix**: A heatmap visualization showing the counts of true versus predicted labels for each digit, which helps identify specific misclassifications.

## 5. Results

The Logistic Regression model achieved the following performance on the test set:

* **Logistic Regression Accuracy**: `0.9528`

The classification report provides a detailed breakdown:
           precision    recall  f1-score   support

       0       1.00      1.00      1.00        29
       1       0.91      1.00      0.95        31
       2       0.95      1.00      0.97        36
       3       0.96      0.86      0.91        29
       4       0.96      1.00      0.98        26
       5       0.92      0.95      0.94        38
       6       0.97      0.94      0.96        34
       7       0.98      0.98      0.98        45
       8       0.88      0.92      0.90        39
       9       1.00      0.91      0.95        53

accuracy                           0.95       360
macro avg       0.95      0.96      0.95       360
weighted avg       0.95      0.95      0.95       360
The confusion matrix visually confirms the model's performance, showing high counts along the diagonal (correct predictions) and low counts off-diagonal (misclassifications).

## 6. Visualizations

The project includes several visualizations to aid in understanding the data and model performance:

* **Sample Digits from the Dataset**: A 5x5 grid of 25 sample handwritten digits, each labeled with its true class.
* **Distribution of Digits (Classes)**: A bar plot showing the frequency of each digit (0-9) in the dataset, indicating a relatively balanced distribution.
* **Distribution of All Pixel Intensity Values**: A histogram illustrating the distribution of pixel values across all images, helping to understand the range and commonality of intensities.
* **Correlation Heatmap of Pixel Intensities**: A heatmap showing the pairwise correlation between the 64 pixel features. This can highlight how different parts of the image relate to each other.
* **2D PCA of Digits Dataset**: A scatter plot of the data projected onto its first two principal components. Points are colored by their digit class, providing an insight into the linear separability of the classes in a reduced dimension.
* **Confusion Matrix - Logistic Regression**: A heatmap displaying the confusion matrix, where rows represent true labels and columns represent predicted labels. Annotated cells show the count of instances, allowing for easy identification of classification errors.

## 7. How to Run

To execute this Jupyter Notebook:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone [https://github.com/Suchendra13/Digits_ML_classify_log_reg.git](https://github.com/Suchendra13/Digits_ML_classify_log_reg.git)
    cd Digits_ML_classify_log_reg
    ```
2.  **Ensure you have Jupyter Notebook installed** or use a compatible IDE (e.g., VS Code with Jupyter extensions).
3.  **Install the required Python libraries**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook Toy_dataset_03.ipynb
    ```
5.  **Run all cells** in the notebook.

## 8. Libraries Used

* `pandas`
* `numpy`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn` (specifically `datasets`, `model_selection`, `preprocessing`, `decomposition`, `linear_model`, `metrics`)

## 9. Acknowledgements

* The `load_digits` dataset is provided by scikit-learn.
