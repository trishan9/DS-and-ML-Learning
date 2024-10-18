___
## Day 16
### Topic: ElasticNet Regression & SVR Deepdive
*Date: October 18, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding ElasticNet Regression and its comparison with Ridge and Lasso
- Learning about Support Vector Regression (SVR)
- Exploring model evaluation metrics (R2 score, MSE, RMSE)
- Practicing Basic hyperparameter tuning using GridSearchCV

### Detailed Notes 📝

### ElasticNet Regression
- Combines L1 (Lasso) and L2 (Ridge) regularization
- Balances feature selection (Lasso) and handling correlated predictors (Ridge)
- Formula: minimizes (RSS + α * [(1-ρ) * L2 + ρ * L1])
  - RSS: Residual Sum of Squares
  - α: regularization strength
  - ρ: balance between L1 and L2 (0 ≤ ρ ≤ 1)

### ElasticNet Summary from the [Paper](https://bmcproc.biomedcentral.com/articles/10.1186/1753-6561-6-S2-S10)

The paper discusses **genomic selection (GS)**, a method used in plant and animal breeding to predict breeding values based on genetic information. Accurate predictions help breeders make better choices about which plants or animals to select for breeding.

**The Importance of Regression Methods**

To predict breeding values, researchers use different statistical techniques. The paper evaluates three regularized linear regression methods: **Ridge Regression**, **Lasso Regression**, and **Elastic Net Regression**. These methods help manage datasets where the number of predictors (variables) is much larger than the number of observations (data points), which is common in genomics.

**1. Ridge Regression**

- **What It Is:** Ridge regression adds a penalty to the size of the coefficients (the numbers that represent the effect of each predictor). This penalty helps to keep the model stable and prevents the coefficients from becoming too large when there are many predictors.
- **How It Works:** It minimizes the error of the prediction while also trying to keep the coefficients small by adding a penalty based on their squares.
- **Strengths:** It works well when there are many predictors that are correlated with each other. Instead of choosing some predictors and ignoring others, it shrinks all the coefficients toward zero but doesn't eliminate them.
- **Weaknesses:** It doesn’t automatically select the most important predictors; all are included in the model.

**2. Lasso Regression**

- **What It Is:** Lasso regression also adds a penalty, but this one is based on the absolute values of the coefficients. This characteristic encourages the model to shrink some coefficients to exactly zero, effectively selecting a simpler model.
- **How It Works:** It minimizes the prediction error while adding a penalty based on the absolute values of the coefficients. This causes some coefficients to become zero, which means those predictors are excluded from the model.
- **Strengths:** It automatically selects relevant predictors, making it easier to interpret the model. This is especially useful in high-dimensional datasets, like those in genomic studies.
- **Weaknesses:** If the predictors are highly correlated, lasso tends to pick one and ignore the others, which might not always be the best choice.

**3. Elastic Net Regression**

- **What It Is:** Elastic Net combines the strengths of both Ridge and Lasso. It uses both types of penalties, which allows it to handle correlations among predictors better than Lasso alone.
- **How It Works:** It minimizes prediction error with penalties based on both the squares (Ridge) and absolute values (Lasso) of the coefficients. This means it can shrink coefficients and also set some to zero.
- **Strengths:** It performs well when predictors are highly correlated and is better at selecting groups of correlated variables. This makes it more flexible and robust in genomic selection.
- **Weaknesses:** While it can select groups of variables, it does not guarantee optimal model selection like some other methods.

**Key Findings from the Paper**

![image](https://github.com/user-attachments/assets/2121eb22-7b56-4a3f-81de-e5b18dd89e2b)

- **Performance:** The paper found that Elastic Net, Lasso, and Adaptive Lasso had similar accuracy levels, and they performed better than Ridge Regression and Ridge Regression BLUP. This means that when trying to predict breeding values using genomic data, Lasso and Elastic Net methods were more effective.
- **Practical Implications:** For breeders, using these more advanced regression techniques can lead to better predictions of breeding values, helping them select the best candidates for breeding. This can improve the efficiency of breeding programs.

### ElasticNet Implementation in Python
![image](https://github.com/user-attachments/assets/1631f301-bab0-4b29-a118-173ebe3a54c8)


### Support Vector Regression
Support Vector Regression (SVR) is a type of machine learning algorithm used for regression tasks. It builds upon the principles of Support Vector Machines (SVM), which are primarily known for classification tasks, but can also be adapted for predicting continuous outcomes, which is the goal of regression.

![1_F0SvFUJxql-H1hYW0j57eA](https://github.com/user-attachments/assets/5b2d4f2c-6b94-46e1-8d79-1bfd82a42632)

![image](https://github.com/user-attachments/assets/45b46aa1-c1a8-47ca-a2dc-99bbbe8c0a34)

### Insights from the [Paper](https://core.ac.uk/download/pdf/81523322.pdf):

#### **Support Vectors**
- In SVR, support vectors are the data points that are closest to the predicted regression line (or hyperplane) but do not fall within the margin of the insensitive tube. They play a crucial role in determining the position of the regression line. Think of them as the critical data points that help define the boundary of the prediction.
- These are the data points that are most important in defining the model. They lie closest to the predicted function and help shape it.

**Example:** If you imagine plotting house prices against their sizes, the points closest to the line (the predicted prices) that help define how the line should slope are your support vectors.

#### **Insensitive Tube (Epsilon Tube)**
- SVR introduces a concept called the "e-insensitive tube." This is a margin around the predicted function where errors (the differences between predicted and actual values) are not penalized if they fall within a certain range. This means that small errors are ignored, making the model robust against noise in the data.
- Insensitive tube, often referred to as the epsilon tube (ε-tube), is a zone around the predicted regression line where errors are ignored. This means if predictions fall within this tube, they do not count as errors. The width of the tube is defined by the epsilon parameter (ε).

**Use Case:** If you're predicting house prices, you might say that any prediction within $5,000 of the actual price is acceptable. So, if the actual price is $300,000, as long as the predicted price is between $295,000 and $305,000, it's considered good.

#### **Kernel Trick**
SVR can use a technique called the kernel trick to handle non-linear data. This means it can transform the input data into a higher-dimensional space where a linear regression line can fit more appropriately. Different types of kernels (like polynomial, radial basis function) can be used depending on the problem.

**Example:** If you were trying to predict house prices and your data shows a non-linear relationship (like prices increasing at an increasing rate as size increases), a kernel trick could help find a better fit.

#### How SVR Works

1. **Training Phase**: The SVR algorithm takes the training data and attempts to find the best-fit line (or hyperplane in higher dimensions) within the epsilon tube. It identifies the support vectors and the parameters ε and C (regularization parameter that balances the trade-off between maximizing the margin and minimizing the prediction error).

2. **Prediction Phase**: For new data, SVR uses the learned regression line to predict values. If the predicted value falls within the ε-tube, it is considered an acceptable prediction; if it falls outside, it is subject to slack variables and may incur a penalty in the model training.

#### Advantages of SVR
- Flexibility: SVR can handle both linear and non-linear relationships through its kernel functions.
- Robustness: The e-insensitive loss function makes it less sensitive to outliers or noisy data points.
- Generalization: SVR generally performs well on unseen data due to its ability to find a balance between fitting the training data and maintaining simplicity.

#### Summary

In summary, Support Vector Regression is a powerful technique for regression tasks that focuses on finding a balance between prediction accuracy and complexity. By utilizing concepts like support vectors, insensitive tubes, slack variables, and the kernel trick, it allows for effective modeling of both linear and non-linear relationships.

### SVR Implementation in Python
![image](https://github.com/user-attachments/assets/443f758e-3132-444f-b725-bba335b33d44)

### Key Takeaways 🔑
- ElasticNet combines strengths of Lasso and Ridge, suitable for correlated predictors
- SVR extends SVM concepts to regression tasks, handling non-linear relationships
- Proper model evaluation and hyperparameter tuning are crucial for optimal performance
- R2, MSE, and RMSE provide different perspectives on model accuracy
- GridSearchCV automates the process of finding the best hyperparameters
