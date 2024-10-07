___
## Day 2
### Topic: Gradient Descent for Linear Regression
*Date: October 4, 2024*

### Today's Learning Objectives Completed ✅
- Understanding Gradient Descent algorithm
- Learning rate and convergence concepts
- Implementation in Python
- Making predictions with optimized parameters

### Detailed Notes 📝

#### Gradient Descent Algorithm
![swappy-20241004_085011](https://github.com/user-attachments/assets/9cc47add-4ec3-4944-b3cc-b8d908902a0e)


The algorithm:
Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving towards the steepest descent as defined by the negative of the gradient. It is widely used in machine learning, especially for minimizing cost functions in algorithms like linear regression and logistic regression.

In machine learning, the goal is often to minimize a cost function to get the optimal set of model parameters (like `w, b` in linear regression). Gradient Descent finds these optimal parameters by iteratively updating them in the direction where the cost decreases most rapidly.
- Initialize the parameters (w and b) to some random values.
- Calculate the cost function for the current parameters.
- Compute the gradient of the cost function with respect to the parameters.
- Simultaneously update the parameters using the gradient to reduce the cost function.
- Repeat this process until the cost function converges (i.e., no significant change in cost).
- Formula:
  ```
  w = w - α * (∂/∂w)J(w,b)
  b = b - α * (∂/∂b)J(w,b)
  ```
  - α (alpha): Learning rate (hyperparameter that controls the size of the steps we take in the direction of the gradient)
  - ∂/∂w, ∂/∂b: Partial derivatives of cost function (controls the direction : where to take step, either left or right)


#### Gradient Descent Intuition
![swappy-20241004_085059](https://github.com/user-attachments/assets/c414b2a7-fe2c-4226-9df0-c6bbe7587852)


- When slope is positive (>0):
  - w decreases (moves left)
  - w = w - α * (positive number)
- When slope is negative (<0):
  - w increases (moves right)
  - w = w - α * (negative number)
- Algorithm naturally moves toward minimum

#### Learning Rate (α) Considerations
![swappy-20241004_085233](https://github.com/user-attachments/assets/55ae693d-19d5-4a31-97b3-b29d3c384162)


Critical aspects:
- If α is too small:
  - Gradient descent will be slow
  - Takes many iterations to converge
- If α is too large:
  - May overshoot the minimum
  - Might fail to converge or even diverge
- Need to choose appropriate learning rate

#### Partial Derivatives (Mathematical Detail)
![swappy-20241004_085357](https://github.com/user-attachments/assets/50348dc6-dff0-4de6-812d-3c1d2e55964f)


Derivatives for batch gradient descent:
```
∂/∂w J(w,b) = (1/m) ∑(fw,b(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾
∂/∂b J(w,b) = (1/m) ∑(fw,b(x⁽ⁱ⁾) - y⁽ⁱ⁾)
```

#### Implementation Results
![swappy-20241004_221626](https://github.com/user-attachments/assets/22041367-a0a1-40fb-b019-a27e3b75b947)


Successfully implemented gradient descent:
- Cost function converged after ~10000 iterations (in plot only first 100 cost history is visualized)
- Final parameters: w ≈ 200, b ≈ 100
- Sample predictions:
  - 1000 sqft house: $300,000
  - 1200 sqft house: $340,000
  - 2000 sqft house: $500,000

#### Key Takeaways 🔑
1. Gradient descent is an iterative optimization algorithm
2. Learning rate is crucial for successful convergence
3. Must update parameters simultaneously
4. Batch gradient descent uses all training examples in comparsion to Stochastic & Mini-Batch
5. Visualization of cost function helps track convergence
