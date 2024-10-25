___
## Day 23
### Topic: KNN, Cross-Validation, Hyperparameter Tuning, Model Evaluation Techniques, Pipelining
*Date: October 25, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding K-Nearest Neighbors (KNN) Algorithm
- Cross-Validation Techniques
- Classification & Regression Evaluation Metrics
- Hyperparameter Tuning Methods
- Pipeline Construction

### Detailed Notes 📝

### K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) is one of the simplest and most intuitive machine learning algorithms. It's a **supervised learning algorithm** that can be used for **classification** tasks. Here's how it works:

#### How KNN works:
- The algorithm doesn't actually "learn" anything during training. Instead, it stores the training data and makes decisions by looking at the data during prediction.
- For a new data point that we want to classify or predict, KNN finds the **K nearest points** in the training set.
- These "nearest points" are found based on a distance metric like **Euclidean distance** (the straight-line distance between two points in space).
 - It checks the **class labels** of the K closest data points and assigns the most common class label (majority vote) to the new point.

#### Example:
Imagine you're trying to classify a fruit as either an apple or a banana. You collect some features like color and size, and plot them on a graph. Now, when you encounter a new fruit that you want to classify, KNN looks at the closest data points (fruits) on the graph and classifies your new fruit based on its "neighbors."

#### Choosing K:
- A **small K** (e.g., K = 1) can lead to **overfitting**, where the model is too sensitive to noise in the training data.
- A **large K** (e.g., K = 50) can lead to **underfitting**, where the model is too simplistic and doesn't capture enough details from the data.

![image](https://github.com/user-attachments/assets/12c7c7d7-715a-4ea0-92da-005dbd2959a0)

---

### **Cross-Validation**
Cross-validation is a technique used to evaluate the performance of a machine learning model. The idea is to avoid overfitting and ensure the model generalizes well to unseen data.

#### How Cross-Validation Works:
- The most common form is **K-Fold Cross-Validation**.
- Here, the dataset is split into K equal-sized parts, called "folds".
- The model is trained on K-1 of the folds and tested on the remaining fold.
- This process is repeated K times, with each fold being used as the test set once.
- Finally, the performance (accuracy, precision, etc.) is averaged over all K runs to get a better estimate of the model's general performance.

#### Example (K-Fold Cross-Validation):
If you have a dataset of 100 samples and use 5-fold cross-validation, the data is divided into 5 parts of 20 samples each. The model trains on 80 samples and tests on 20 samples, repeating this 5 times so each fold acts as the test set once. Then you average the performance metrics (like accuracy) across all 5 runs.

---

### Evaluation Metrics for Classification

When evaluating classification models, a range of metrics assess how well the model distinguishes between classes. Let’s discuss the key metrics in depth:

1. **Confusion Matrix**:
   - **Definition**: A table showing counts for True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
   - **Example**: For a model that predicts if an email is spam:
     - **TP**: Model correctly identifies a spam email.
     - **TN**: Model correctly identifies a non-spam email.
     - **FP**: Model incorrectly labels a non-spam email as spam.
     - **FN**: Model incorrectly labels a spam email as non-spam.


   |               | Predicted Spam | Predicted Not Spam |
   |---------------|----------------|--------------------|
   | **Actual Spam**     | TP             | FN                 |
   | **Actual Not Spam** | FP             | TN                 |

2. **Accuracy**:
   - **Formula**: (TP + TN)/(TP + TN + FP + FN)

   - **Explanation**: Measures the proportion of correct predictions. It’s best used when the classes are balanced, as it can be misleading in imbalanced datasets.

3. **Precision**:
   - **Formula**: TP/(TP + FP)
   - **Explanation**: Precision represents how many of the positive predictions are truly positive. High precision means fewer false positives.
   - **Example**: In spam detection, a high precision means most emails flagged as spam are actually spam.

4. **Recall**:
   - **Formula**: TP/(TP + FN)
   - **Explanation**: Recall (also known as sensitivity or true positive rate) measures the ability to capture all actual positives. High recall means fewer false negatives.
   - **Example**: In spam detection, high recall means that most spam emails are detected as spam.

5. **F1 Score**:
   - **Formula**: 2 × (Precision × Recall)/(Precision + Recall)
   - **Explanation**: The F1 Score is a harmonic mean of precision and recall, balancing both metrics. It’s especially useful in cases of imbalanced data where precision and recall are both important.
   - **Example**: If a spam detector has high recall but low precision, the F1 Score will indicate a balanced view of its performance.

6. **ROC Curve and AUC (for Binary Classification)**:
   - **ROC Curve**: Plots the **True Positive Rate** (Recall) against the **False Positive Rate** across different threshold values.
   - **AUC (Area Under Curve)**: Represents the model’s capability to rank positive samples higher than negative ones. A higher AUC (close to 1) indicates better performance.
   - **Example**: In medical diagnosis, an AUC of 0.9 means the model is 90% likely to rank a positive case higher than a negative one.

---

### **Evaluation Metrics for Regression**

In regression problems, metrics quantify the difference between the predicted and actual values.

1. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values, making it easy to interpret in the same units as the target variable.

2. **Mean Squared Error (MSE)**: Penalizes larger errors more heavily by squaring them, which can be useful if large errors are particularly undesirable.

3. **Root Mean Squared Error (RMSE)**: Often easier to interpret than MSE, RMSE has the same units as the target variable and is useful for assessing the standard deviation of errors.

4. **R-squared (R²)**: Represents the proportion of variance explained by the model, where a score closer to 1 indicates a better fit.

---

### **Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV**

Hyperparameter tuning involves optimizing settings that are not learned from data (like `k` in KNN, max depth in decision trees). Two popular methods for tuning:

- **GridSearchCV:** Exhaustively tests a range of values for each hyperparameter, creating every possible combination and training the model on each.
- **RandomizedSearchCV:** Randomly selects combinations of hyperparameters for testing, which is faster and often effective for large search spaces.

---

### **Using `cross_val_score` to Analyze Model Performance**

`cross_val_score` allows you to run cross-validation on different models to assess which performs best on your dataset. It takes in a model and returns the performance score for each fold, which can then be averaged to evaluate stability and effectiveness across folds.

---

### **Pipelines for Efficient Workflow**

A pipeline lets you streamline data preprocessing (like imputation and scaling) and model training in a single, organized workflow.

**Example:** For a classification task, you can create a pipeline to:
1. Fill in missing values with an imputer.
2. Scale features using StandardScaler.
3. Pass the data into a KNN or Logistic Regression model.

Each of these steps (imputing, scaling, and model selection) can include hyperparameter tuning within the pipeline itself, making the entire process efficient and less error-prone.

### Implementation
**Simple KNN with pipelines, and hyperparameter tuning**

![image](https://github.com/user-attachments/assets/54d7fcce-4526-426c-926e-c86af6ce274a)

**Classification model with analysing accuracy scores for different models**

![image](https://github.com/user-attachments/assets/3bb860ff-3dc5-4551-9bbe-10eab9435cf4)
![image](https://github.com/user-attachments/assets/1410f048-0ebd-4edf-9c69-c2fb0c4e05aa)
