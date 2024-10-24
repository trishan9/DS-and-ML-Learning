___
## Day 22
### Topic: Deepdive and Implementation of Logistic Regression
*Date: October 24, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding Maximum Likelihood Estimation (MLE)
- Regularization in Logistic Regression
- Practical Implementation with sklearn
- Real-world Applications and Evaluation Metrics, Cross-validation Techniques

### Detailed Notes 📝

### Maximum Likelihood and Regularized Logistic Regression

In logistic regression, the model estimates probabilities using the sigmoid function, and the process of estimating the model parameters (`w` and `b`) can be viewed as a Maximum Likelihood Estimation (MLE) problem. The goal is to find the parameters that maximize the likelihood of the observed data given the model. The cost function, also known as the **log-likelihood** function, for logistic regression is minimized to achieve this.

![image](https://github.com/user-attachments/assets/21301db0-b296-4a2b-9da6-686515150196)


However, when dealing with many features or complex models (like high-order polynomials), logistic regression can suffer from **overfitting**, which means it performs well on training data but poorly on unseen data. To address this, **regularization** is introduced.

**Regularized Logistic Regression** adds a penalty to the cost function to discourage overly complex models. The modified cost function becomes:

J(w, b) = existing cost function + λ/2m ∑(wⱼ²)

The second term, is the **regularization term** (L2 regularization) that penalizes large weights, which helps prevent overfitting by keeping the weights small. The gradient descent update rule is modified to account for this term, ensuring that the parameters are updated in a way that balances both fitting the data well and avoiding complex decision boundaries.

![image](https://github.com/user-attachments/assets/66c73132-a0d4-4bfb-bcdc-d224b63d7a7b)


### Python Implementation of Logistic Regression on the Iris Dataset

![image](https://github.com/user-attachments/assets/a3dd7133-673e-4230-a089-b432b43db976)


- **Dataset**: The Iris dataset was used, but for simplicity, I performed **binary classification** by removing one class (i.e., focusing on just two classes).
- **Model**: I used the `LogisticRegression` model from `sklearn` and trained it on the training set.
- **Evaluation**: I evaluated the model using metrics like **accuracy**, **precision**, **recall**, **F1 score**, and the **confusion matrix**.
- **Key Result**: The predictions and the corresponding performance metrics (accuracy, precision, recall, and F1 score) help determine how well the model performs on the test data.

### Logistic Regression on the Breast Cancer Dataset (Real-World Example)

![image](https://github.com/user-attachments/assets/810429ae-2729-4575-9c8d-0aebae1e3aa1)


- **Dataset**: The breast cancer dataset has multiple features (over 9), and I tried to classify whether a tumor is benign or malignant.
- **Model**: I used `LogisticRegression` from `sklearn` to fit the model on the training data.
- **Confusion Matrix**: A confusion matrix was printed to evaluate how many true positives, false positives, true negatives, and false negatives were predicted by the model.
- **Cross-Validation**: I applied **10-fold cross-validation** to get a better estimate of the model’s performance. This is useful because it gives insight into how the model generalizes across different splits of the dataset, and also provides an estimate of performance variability (standard deviation).

Both implementations of **Logistic Regression** in Python show how the algorithm can be applied to real-world problems, like binary classification in the Iris dataset and breast cancer classification. Regularization, if implemented, would have helped avoid overfitting, especially in the breast cancer dataset with many features. Through metrics like **accuracy**, **precision**, **recall**, and **cross-validation**, I’ve ensured that the model performs well on unseen data.
