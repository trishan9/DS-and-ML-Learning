___
## Day 12
### Topic: Polynomial Regression
*Date: October 14, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding the need for Polynomial Regression
- Mathematical foundations of Polynomial Regression
- How Polynomial Regression works
- Intuition behind Polynomial Regression
- Example and practical considerations

### Detailed Notes 📝

### Why Polynomial Regression?

![image](https://github.com/user-attachments/assets/592c4619-8ce0-4d70-b0ec-da117ac88679)

#### Linear Regression Limitations
- Linear regression assumes a straight-line relationship between features and target
- Real-world data often has non-linear patterns

#### Non-linear Relationships
- Polynomial regression is useful when data points form a curve that cannot be adequately described by a linear model.
- It helps to capture the turning points, bends, and curves in the data.

#### Model Flexibility
- Adding polynomial terms allows for more flexible adaptation to data trends

### Mathematics Behind Polynomial Regression

![image](https://github.com/user-attachments/assets/3359fc29-9ada-4315-b15c-2375fe16fad3)


Polynomial equation:
Polynomial regression is a special case of multiple linear regression because we still use a linear equation, but with polynomial features. The polynomial equation is given by:

y = β₀ + β₁x + β₂x² + β₃x³ + ⋯ + βₙxⁿ

Where:
- y: dependent variable
- x: independent variable
- β₀, β₁, β₂, ..., βₙ: coefficients
- n: degree of the polynomial

### How Does It Work?

- Transforms original features into higher-degree polynomial features
- Applies linear regression to fit these transformed features
- Example: x → x, x², x³, ..., xⁿ

### Intuition Behind Polynomial Regression

- The idea behind polynomial regression is that by adding higher-degree terms, we can better fit the training data by capturing more of its variations.
- **Low Degree**: Underfitting might occur if the polynomial degree is too low to capture the data's trends.
- **High Degree**: Overfitting might occur if the polynomial degree is too high, causing the model to fit the training data too well, including the noise.

### Example of Polynomial Regression

Step-by-step Example:
1. **Original Data**: Let's say our data points look like a quadratic curve.
2. **Choosing the Polynomial Degree**: We choose a degree of 2, which means the equation becomes:
   y = β₀ + β₁x + β₂x²
3. **Fitting the Model**: We fit this quadratic equation to our data using linear regression techniques.
4. **Prediction**: We can now make predictions using the fitted polynomial curve.

#### Underfitting and Overfitting:
- **Underfitting**: If the degree of the polynomial is too low, the model will not capture the true trend in the data.
- **Overfitting**: If the degree is too high, the model will fit the noise in the training data and perform poorly on unseen data.

#### Choosing the Right Degree
- Experiment with different degrees of the polynomial to find the best fit.
- Use metrics like Mean Squared Error (MSE) or R-squared to compare model performance.

### Implementation of Polynomial Regression
**Python**

![image](https://github.com/user-attachments/assets/2f90794a-8804-49c6-b853-31b4f97ebb4c)
![image](https://github.com/user-attachments/assets/73bd1480-d88d-49a8-a5a7-259123d6ca16)

**R**

![image](https://github.com/user-attachments/assets/6f283cab-2d14-4988-b3e4-5d26c41cf2e2)
![image](https://github.com/user-attachments/assets/ffe99ca7-242b-4ec7-bd58-1e5a2e1844d6)


### Key Insights 🔑
- Polynomial regression captures non-linear trends in data
- It's an extension of linear regression with higher-degree terms
- Proper degree selection is crucial to avoid under/overfitting
- Visual representation aids in understanding model performance
- Balancing complexity and generalization is key to a good model
