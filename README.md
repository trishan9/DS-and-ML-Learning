# DS-and-ML-Learning
___

This repo consists of my whole Data Science and Machine Learning journey and here I will be documenting my complete journey! Inspired from: [**iamshishirbhattarai/100DaysOfMachineLearning**](https://github.com/iamshishirbhattarai/100DaysOfMachineLearning)
___
## Syllabus
This is just a pre-setup and things are added as exploration continues !!

| **S.N.** | **Books and Lessons (Resources)**                                                                                                 | **Status** |
|----------|-----------------------------------------------------------------------------------------------------------------------------------|------------| 
| **1.**   | [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)          | ⏳          |
| **2.**   | [**Machine Learning Scientist With Python**](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) | ⏳          |
| **3.**   | [**Associate Data Scientist in Python**](https://app.datacamp.com/learn/career-tracks/associate-data-scientist-in-python) | ⏳          |
| **4.**   | [**Mathematics for Machine Learning and Data Science Specialization**](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science) | ⏳          |
| **5.**   | [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/) | ⏳          |

___

## Projects Completed

| **S.N.** | **Project Title**                                                                                                                                                                                | **Status** |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| 1.       |  |           |



## Topics Learnt

| **Days**        | **Learnt Topics**                                                                                                                                                            | **Resources used**                                                                                                                                                                                                                                  |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Day 1](Day1)   |         Linear Regression, Cost Function and Optimization                                                                                                       |          [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)                                                                                                                                |



___
## Day 1 
### Topic: Linear Regression Fundamentals
*Date: October 3, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding Linear Regression
- Mathematical foundations
- Cost function and optimization
- Model parameters and fitting

### Detailed Notes 📝

**Linear Regression Model Basics**

- Linear regression is a supervised learning algorithm for predicting a continuous output variable (y) based on input features (x)
- The model represents a linear relationship: `f(x) = wx + b`
  - w: weight/slope
  - b: bias/y-intercept
- Use case: Predicting numerical values (e.g., house prices, stock prices, temperature forecasting)

**Model Components**

![image](https://github.com/user-attachments/assets/a874b11c-c6ed-44a6-8956-bb1127aa8ab9)

- Training set: Collection of input-output pairs for learning
- Features (x): Input variables
- Targets (y): Actual output values
- Predictions (ŷ): Model's estimated outputs
- Model function: f(x) = wx + b

**Mathematical Framework & Cost Function**

![image](https://github.com/user-attachments/assets/9d6822c0-0866-46b7-8d09-519d6eec4863)

- Model equation: **f<sub>w,b</sub>(x) = wx + b**
- Parameters:
  - w (weight): Determines slope
  - b (bias): Determines y-intercept
- Simplified version: **f<sub>w</sub>(x) = wx (when b = 0)** (just for learning, as it can be visualized in 2D easily)
- Cost Function: Measures how well our model fits the data (quantifies the error between predictions and actual results.)
- Squared Error Cost Function: **J(w,b) = (1/2m) ∑(f<sub>w,b</sub>(x⁽ⁱ⁾) - y⁽ⁱ⁾)²** where,
  - m: number of training examples
  - x⁽ⁱ⁾: i-th input feature
  - y⁽ⁱ⁾: i-th actual output
 
**Optimization Goal (Minimizing the Cost Function)**

![image](https://github.com/user-attachments/assets/5de4831b-78a2-4a33-954c-a808c5a29bf7)

- Objective: Find best values of w and b that minimize J(w,b) which tells us how well our linear model fits the data.
- The optimal point occurs where:
  - Cost function J(w) reaches its minimum
  - In the example graph, w ≈ 1 gives minimum cost

**Visual Intuition**

![image](https://github.com/user-attachments/assets/6fd68e1f-a34f-4a89-83d9-ad912aff7fba)

- Cost function forms a soup bowl-shaped surface in 3D
- Global minimum exists at the bottom of the bowl
- Goal is to find the coordinates (w,b) at this minimum

**Key Takeaways 🔑**

- Linear regression finds a linear relationship between input and output
- Model is represented by f(x) = wx + b
- Cost function measures prediction errors
- Goal is to minimize cost function by finding optimal w and b
- Visualization helps understand the optimization landscape

**Tomorrow's Goals 🎯**

- Dive into gradient descent algorithm
- Implement linear regression from scratch
- Practice with real-world dataset

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

#### Tomorrow's Goals 🎯
- Multi-feature linear regression
- Vectorization
- Gradient descent for multiple linear regression
- Feature Scaling, Feature Engineering, Polynomial Regression
- Practice implementing with scikit-learn
