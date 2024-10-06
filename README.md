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
| [Day 2](Day2)   |         Gradient Descent Algorithm, Intuition, Implementation                                                                                                       |          [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)                                                                                                                                |
| [Day 3](Day3)   |         Multiple Feature Linear Regression                                                                                                       |          [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)                                                                                                                                |
| [Day 4](Day4)   |         ML Process and Data Pre-processing                                                                                                       |          [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |


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

___
## Day 3
### Topic: Multiple Linear Regression
*Date: October 5, 2024*

### Today's Learning Objectives Completed ✅
- Understanding Multiple Feature Linear Regression
- Vector notation in Linear Regression
- Feature representation and indexing
- Extended model equations

### Detailed Notes 📝

#### Multiple Features Introduction
![swappy-20241005_214718](https://github.com/user-attachments/assets/ae8f424d-5ad9-420a-aef7-e6a1bab277cf)


Important notation:
- n = number of features
- m = number of training examples
- x<sup>(i)</sup> = features of i<sup>th</sup> training example
- x<sub>j</sub><sup>(i)</sup> = value of feature j in i<sup>th</sup> training example

Example from the data:
- x<sup>(2)</sup> = [1416  3  2  40] (complete 2nd training example)
- x<sub>3</sub><sup>(2)</sup> = 2 (3rd feature of 2nd training example)

#### Model Extension
![swappy-20241005_214726](https://github.com/user-attachments/assets/8e3e9689-8c16-49a4-8e0d-ecd69540afb3)


Evolution of the model:
- Previously: f<sub>w,b</sub>(x) = wx + b
- Now with multiple features:
  ```
  fw,b(x) = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b
  ```

Example house price prediction:
```
fw,b(x) = 0.1x₁ + 4x₂ + 10x₃ - 2x₄ + 80
```
where:
- x₁: size in feet²
- x₂: number of bedrooms
- x₃: number of floors
- x₄: age of home in years
- b = 80: base price

#### Vector Notation
![swappy-20241005_214740](https://github.com/user-attachments/assets/eadb2826-0e00-4101-8f95-b40e344b7966)


Modern representation using vectors:
- w⃗ = [w₁ w₂ w₃ ... wₙ] (parameter vector)
- x⃗ = [x₁ x₂ x₃ ... xₙ] (feature vector)
- b is a single number (scalar)

Final model equation using dot product:
```
fw,b(x) = w⃗ · x⃗ + b = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b
```

**Important Note**: This is multiple linear regression, not multivariate regression. The distinction is that we have multiple features (variables) but still predict a single output value.

#### Key Takeaways 🔑
1. Multiple features allow more complex and accurate predictions
2. Vector notation simplifies representation of multiple features
3. Dot product provides elegant mathematical formulation
4. Each feature has its own weight parameter (w)
5. Base price (b) remains a single scalar value

#### Practical Implementation Tips 💡
- Use vectors and matrices for efficient computation
- Keep track of feature indices carefully
- Document feature meanings and units
- Consider feature scaling for better performance
- Use proper indexing notation in code

___
## Day 4
### Topic: Machine Learning Process and Data Pre-processing
*Date: October 6, 2024*

### Today's Learning Objectives Completed ✅
- Understanding the machine learning process
- Data Pre-processing theory
- Data Pre-processing implementation in Python using Scikit-learn
- Concepts of encoding and handling missing data
- Splitting dataset into training and test sets
- Feature scaling: understanding types and necessity
- Implementation of data pre-processing in Python

### Detailed Notes 📝

#### The Machine Learning Process Overview
![image](https://github.com/user-attachments/assets/aa31449a-9941-42e7-a683-8459d7d8f9bc)

The machine learning process can be broken down into three main stages:
1. **Data Pre-Processing**: 
    - Import the data
    - Clean the data (handle missing values, encoding categorical data)
    - Split into training and test sets
    - Feature scaling (normalization/standardization)
  
2. **Modelling**: 
    - Build the model
    - Train the model
    - Make predictions
  
3. **Evaluation**: 
    - Calculate performance metrics
    - Make a verdict

#### Data Pre-processing Steps

1. **Importing the Data**: 
    This involves loading the dataset into your Python environment. In my case, I used the Pandas library to import CSV data into a DataFrame.

2. **Handling Missing Data**:
    Missing data can be handled in multiple ways:
    - Removing the missing data rows (not recommended in all cases).
    - Replacing the missing values with mean, median, or most frequent values. I replaced missing data in this case using `SimpleImputer` from Scikit-learn.

3. **Encoding Categorical Data**:
    Categorical variables must be encoded as numerical values for ML algorithms. I used `LabelEncoder` for encoding categorical variables.

4. **Splitting the Dataset**: 
    The dataset is split into a **Training set** (to train the model) and a **Test set** (to evaluate model performance).
   
    ![image](https://github.com/user-attachments/assets/d490577f-72bd-45d1-9d6f-c82b81f42688)


    - I used the `train_test_split` function from Scikit-learn to split the data in an 80:20 ratio for training and testing.

5. **Feature Scaling**:
    Feature scaling ensures that all the features are on the same scale, improving the performance of machine learning models.

   ![image](https://github.com/user-attachments/assets/930c2fa0-804b-49e9-b89f-5c5d558b76d9)


    There are two types of feature scaling:
    - **Normalization**: Scales values between 0 and 1.
    - **Standardization**: Scales data with a mean of 0 and standard deviation of 1.
      
    ![image](https://github.com/user-attachments/assets/bf5505f9-de35-45fb-9105-1f57c89aaef5)


#### Python Implementation 🖥️
I implemented Data Pre-processing in Python using scikit-learn:

![image](https://github.com/user-attachments/assets/b603e2bf-b2f8-4591-9a1e-3df3529177dd)


#### Key Takeaways 🔑
- Pre-processing is crucial for ensuring that data is clean and well-prepared for training.
- Handling missing data can have a significant impact on model performance.
- Encoding categorical data is necessary to convert text labels into a format that machine learning models can understand.
- Feature scaling ensures that all features contribute equally to the model's learning process.
- Splitting data ensures that model evaluation is performed on unseen data, preventing overfitting.

#### Some diagrams
**ML Process Flow**


![image](https://github.com/user-attachments/assets/3a97d53b-d756-43be-8129-838aee56a028)


**Dataset Splitting and Scaling Process**


![image](https://github.com/user-attachments/assets/fbb325d9-9c00-414d-8c65-a3adc9a9cf70)


**Feature Scaling Methods Comparison**


![image](https://github.com/user-attachments/assets/139cdc5c-dec9-4867-8f28-4db9d4cdebee)


**Data Preprocessing Steps**


![image](https://github.com/user-attachments/assets/55396f9c-4a9d-4da9-bc0f-6af35e406a80)
