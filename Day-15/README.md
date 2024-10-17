___
## Day 15
### Topic: Ridge & Lasso Regression Implementation in Python
*Date: October 17, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding Ridge and Lasso Regression
- Implementing Ridge and Lasso Regression using scikit-learn
- Hyperparameter tuning with GridSearchCV
- Evaluating model performance

### Detailed Notes 📝

### Ridge and Lasso Regression
- Both are regularized versions of linear regression
- Help prevent overfitting by penalizing large coefficients
- Implemented using scikit-learn library in Python

#### Ridge Regression
- Uses L2 regularization
- Shrinks coefficients but does not eliminate them
- Helps reduce multicollinearity

#### Lasso Regression
- Uses L1 regularization
- Can drive some coefficients to zero, performing feature selection
- Useful for identifying important features

### Implementation Steps
1. Import necessary libraries (sklearn, numpy)
2. Load and split data
3. Perform hyperparameter tuning using GridSearchCV
4. Train the model with best hyperparameters
5. Evaluate model performance using Mean Squared Error, and Plots

### Key Concepts
- Regularization strength (alpha/lambda): Controls the penalty on model complexity
- GridSearchCV: Automates hyperparameter tuning using cross-validation
- Mean Squared Error (MSE): Measures the average squared difference between predictions and actual values

### Code Snippets
```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Hyperparameter tuning
param_grid = {'alpha': np.logspace(-4, 4, 50)}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Train and evaluate
best_model = model(alpha=best_alpha)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

### Visualizing Results
- KDE (Kernel Density Estimate) plot used to visualize prediction errors
- Helps understand the distribution of residuals

![image](https://github.com/user-attachments/assets/c57077e8-897a-4888-9ad8-ed476b1a6fc8)

### Key Differences between Ridge and Lasso
- Ridge: Shrinks coefficients, doesn't perform feature selection
- Lasso: Can eliminate less important features by setting coefficients to zero

### Key Takeaways 🔑
- Ridge and Lasso are powerful techniques for preventing overfitting in linear regression
- Hyperparameter tuning is crucial for optimal model performance
- Lasso can perform feature selection, while Ridge helps with multicollinearity
- Visualization of residuals provides insights into model performance
- Both methods balance model complexity and accuracy
