___
## Day 14
### Topic: Ridge & Lasso Regression
*Date: October 16, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding Ridge Regression (L2 regularization)
- Exploring Lasso Regression (L1 regularization)
- Comparing Ridge and Lasso Regression techniques
- Learning about Elastic Net
- Analyzing practical insights from regularization techniques

### Detailed Notes 📝

Ridge Regression and Lasso Regression are both types of linear regression techniques used to address multicollinearity and overfitting in machine learning models. They are regularization methods that add a penalty to the regression model to reduce the magnitude of the coefficients.

### Ridge Regression (L2 Regularization)
Ridge Regression, also known as **L2 regularization**, adds a penalty equal to the sum of the squared values of the coefficients (except the intercept) to the cost function. It aims to shrink the coefficients of less important features closer to zero without completely removing them.

- Also known as L2 regularization
- Adds penalty equal to the sum of squared coefficients to the cost function
- Cost Function: RSS + λ ∑(β_j^2)
  - RSS: Residual Sum of Squares
  - λ: Regularization parameter
  - β_j: Coefficients of features

#### Effect of the Penalty

When **λ** is increased, the coefficients of the less significant features shrink towards zero, but they never reach exactly zero. Ridge regression is useful when many predictors are correlated.

#### Impact

Ridge Regression performs well when all the predictor variables are important but may have multicollinearity issues. It helps stabilize the model by reducing variance.

#### Example

In practice, Ridge Regression is used to deal with datasets where features are correlated. For instance, if you have features like "square footage" and "number of rooms" in predicting house prices, Ridge Regression helps to stabilize the model coefficients and prevent overfitting.

![image](https://github.com/user-attachments/assets/d7d1d64d-2ac1-49c7-99ad-42a99348b8d7)
![image](https://github.com/user-attachments/assets/20df953d-57a0-46e0-aab0-2c046fbbb970)

### Lasso Regression (L1 Regularization)
Lasso Regression, short for **Least Absolute Shrinkage and Selection Operator** or **L1 regularization**, adds a penalty equal to the absolute values of the coefficients to the cost function. Unlike Ridge Regression, Lasso Regression can shrink some coefficients to zero, effectively performing feature selection.

- Least Absolute Shrinkage and Selection Operator
- Adds penalty equal to the absolute values of coefficients to the cost function
- Cost Function: RSS + λ ∑|β_j|

#### Effect of the Penalty

Lasso Regression forces some coefficients to become exactly zero when the regularization parameter is large enough. This makes Lasso useful for feature selection, as it automatically excludes irrelevant features from the model.

#### Impact

Lasso Regression works well when there are many predictors, but only a subset of them are actually relevant for predicting the target variable. It helps simplify the model and reduce overfitting.

#### Example

Lasso Regression is particularly useful in scenarios where feature selection is crucial. For example, in genetics, if you have thousands of genetic markers, Lasso can automatically select only the few markers that have a significant impact on the disease risk prediction.

![image](https://github.com/user-attachments/assets/e2e481f6-6d77-49d8-bcb9-4ce5ef49061d)

### Ridge vs Lasso Regression

| **Feature**             | **Ridge Regression (L2)**                    | **Lasso Regression (L1)**                    |
|-------------------------|---------------------------------------------|---------------------------------------------|
| **Penalty Type**        | Squared magnitude of the coefficients        | Absolute magnitude of the coefficients      |
| **Feature Selection**   | Does not perform feature selection           | Can perform feature selection by shrinking coefficients to zero |
| **Coefficient Shrinkage**| Shrinks coefficients towards zero but not exactly zero | Can shrink coefficients exactly to zero    |
| **Use Case**            | When all features are important but might be collinear | When only a few features are important     |

### Choosing Between Ridge and Lasso Regression
- Use **Ridge Regression** when you expect all features to be relevant, but some might be correlated. It helps to control overfitting by penalizing large coefficients.
- Use **Lasso Regression** if you suspect that only a few predictors have significant influence on the response variable. Lasso's ability to perform feature selection can lead to a simpler, more interpretable model.

### Elastic Net
Sometimes, a combination of Ridge and Lasso Regression can be used, called **Elastic Net**. Elastic Net is useful when there are multiple correlated features, as it balances the strengths of both Ridge and Lasso Regression. It combines the penalties of both L1 and L2 regularization:

- Combination of Ridge and Lasso Regression
- Cost Function: RSS + λ1 ∑|β_j| + λ2 ∑(β_j^2)
- Balances strengths of both Ridge and Lasso
- Useful when multiple correlated features exist

### Practical Insights

1. **Ridge Regression**:
   - Useful when all predictors are important.
   - Helps mitigate the issue of multicollinearity.
   - Coefficients never become exactly zero, hence all features are retained in the model.

2. **Lasso Regression**:
   - Ideal for automatic feature selection and simpler models.
   - Forces some coefficients to be exactly zero, eliminating irrelevant features.
   - Suitable when you expect only a subset of features to be impactful.

### Key Insights 🔑
- Ridge and Lasso are regularization techniques to address multicollinearity and overfitting
- Ridge shrinks coefficients towards zero, Lasso can shrink them to exactly zero
- Lasso performs feature selection, while Ridge retains all features
- Elastic Net combines both approaches for balanced regularization
- Choice between Ridge and Lasso depends on the specific problem and data characteristics
- Regularization techniques help in creating more stable and interpretable models
