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
