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
| [Day 1](#day-1)   |         Linear Regression, Cost Function and Optimization                                                                                                       |          [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)                                                                                                                                |
| [Day 2](#day-2)   |         Gradient Descent Algorithm, Intuition, Implementation                                                                                                       |          [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)                                                                                                                                |
| [Day 3](#day-3)   |         Multiple Feature Linear Regression                                                                                                       |          [**Machine Learning Specialization**](https://www.coursera.org/specializations/machine-learning-introduction)                                                                                                                                |
| [Day 4](#day-4)   |         ML Process and Data Pre-processing                                                                                                       |          [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |
| [Day 5](#day-5)   |         Data Pre-processing Techniques in R                                                                                                       |          [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |
| [Day 6](#day-6)   |         Vectorization, Gradient Descent for Multiple Linear Regression                                                                                                       |          [**Machine Learning Specialization**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |
| [Day 7](#day-7)   |         Feature Scaling & Gradient Descent Optimization                                                                                                       |          [**Machine Learning Specialization**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |
| [Day 8](#day-8)   |         Exploratory Data Analysis in Python                                                                                                       |          [**Associate Data Scientist in Python**](https://app.datacamp.com/learn/career-tracks/associate-data-scientist-in-python)                                                                                                                                |
| [Day 9](#day-9)   |         Simple Linear Regression in Python & R                                                                                                       |          [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |

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


___
## Day 5
### Topic: Data Pre-processing Techniques in R
*Date: October 7, 2024*

### Today's Learning Objectives Completed ✅
- Deep dive into data preprocessing concepts
- Implemented data preprocessing in R
- Understanding the importance of handling missing data
- Encoding of categorical data in R
- Importance of feature scaling and its correct application
- Concept of avoiding information leakage by scaling after dataset splitting

### Detailed Notes 📝

#### Advanced Data Pre-processing Techniques

Today, I expanded my understanding of data preprocessing and implemented these techniques using R programming. Here are the detailed concepts and methods I focused on:

1. **Handling Missing Data**:
   - Missing data can create bias and reduce model accuracy.
   - Various strategies to handle missing data include removing missing values, replacing them with mean, median, mode, or more sophisticated techniques.
   - Importance: Ensuring that no important information is lost while maintaining the dataset's integrity.

2. **Encoding Categorical Data**:
   - Machine learning algorithms work better with numerical data; hence, categorical variables need to be converted into numbers.
   - Encoding techniques like One-Hot Encoding and Label Encoding were explored using R functions.
   - In R, I used the `factor()` function for Label Encoding.
   - This step is crucial for ensuring that algorithms can interpret categorical features properly.

3. **Splitting the Dataset into Training and Test Sets**:
   - Splitting the dataset ensures that the model is trained on one part of the data and tested on another, which is crucial for evaluating its performance.
   - In R, I used the `sample.split()` function from the `caTools` package to divide the dataset into training and test sets.
   - Proper splitting prevents the model from overfitting on the training data and ensures a fair evaluation on unseen data.

4. **Feature Scaling**:
   - Feature scaling standardizes the range of independent variables or features of the dataset.
   - Two main methods used:
     - **Normalization**: Scales data between 0 and 1.
     - **Standardization**: Scales data to have a mean of 0 and a standard deviation of 1.
   - Scaling in R was performed using functions like `scale()` for Standardization.

5. **Avoiding Information Leakage**:
   - **Information leakage** occurs when data from the test set influences the model training process, leading to over-optimistic performance estimates.
   - To prevent this, feature scaling should be done **after** splitting the dataset into training and test sets.
   - By scaling only the training set first, we ensure that the test set remains completely unseen during model training, preserving its role as unseen data for evaluation.

#### R Implementation 🖥️

Below is my code snippet for data preprocessing using R:

```R
# Importing Dataset
dataset = read.csv('Data.csv')

# Handling missing data
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN=function(x) mean(x, na.rm=TRUE)), dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN=function(x) mean(x, na.rm=TRUE)), dataset$Salary)

# Encoding Categorical Data
dataset$Country = factor(dataset$Country, levels=c('France', 'Spain', 'Germany'), labels=c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased, levels=c('Yes', 'No'), labels=c(0, 1))

# Splitting the dataset
library(caTools)
set.seed(100)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
```


![image](https://github.com/user-attachments/assets/343b9296-49e8-4fc4-974e-a4369c113b63)


#### Key Takeaways 🔑
- Proper handling of missing data ensures no important information is lost.
- Encoding categorical variables correctly improves model interpretability.
- Splitting the dataset into training and test sets ensures a fair evaluation of model performance.
- Feature scaling should always be done after splitting the dataset to avoid information leakage.
- Implementing data preprocessing in R is efficient and follows a structured approach using native R functions.
- Following best practices in data preprocessing improves the robustness and reliability of machine learning models.


___
## Day 6
### Topic: Vectorization & Multiple Feature Linear Regression
*Date: October 8, 2024*

**Today's Learning Objectives Completed ✅**
- Vectorization in NumPy for Linear Regression
- Efficient implementation using vector operations
- Gradient Descent for multiple features
- Normal Equation as an alternative approach
- Mathematical notation and implementations

### Detailed Notes 📝

#### Vectorization Fundamentals
Explored how vectorization simplifies the code when implementing learning algorithms. It makes the code not only shorter but also significantly more efficient. By leveraging modern numerical linear algebra libraries (like NumPy) and even GPU hardware, vectorized implementations can run much faster compared to unvectorized versions.

Vectorization involves performing operations on entire arrays or matrices, instead of using explicit loops. It allows us to utilize optimized low-level implementations and take advantage of parallelism.

![image](https://github.com/user-attachments/assets/75e1789d-e2e1-420f-9d17-0d3549279e8c)


**Key Components:**
- Parameters represented as vectors:
  - w = [w₁ w₂ w₃] for weights
  - x = [x₁ x₂ x₃] for features
  - b as a scalar bias term
- Non-vectorized implementation uses loops:
  ```python
  f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b
  ```
- Vectorized version uses dot product:
  ```python
  f = np.dot(w,x) + b
  ```

#### Performance Benefits
![image](https://github.com/user-attachments/assets/515d0cf8-f6a4-4790-8322-628da510c800)


**Advantages:**
- Shorter code
- Faster execution
- Parallel computation of element-wise operations
- Efficient memory usage
- Leverages optimized linear algebra libraries
- Scales well with large datasets
- Potential GPU acceleration

**Example of Speed Improvement:**
- Without vectorization: Sequential operations at times t₀, t₁, ..., t₁₅
- With vectorization: Single parallel operation computing all multiplications simultaneously

#### Gradient Descent Implementation
![image](https://github.com/user-attachments/assets/d4fd6482-5e2a-4b51-b97e-5dd3ddb6ef1a)


**Vectorized Updates:**
- Parameters update: w = w - 0.1*d
- Learning rate (α) = 0.1
- Derivatives stored in vector d
- Single operation updates all parameters simultaneously

#### Mathematical Notation
![image](https://github.com/user-attachments/assets/44f3b4ba-dfd8-404b-9ca9-16e3ffcc1c17)


**Improved Notation:**
- Traditional: w₁, w₂, ..., wₙ as separate variables
- Vector notation: w = [w₁ ... wₙ]
- Model function: f(x) = w·x + b
- Simplified gradient descent expressions

#### Multiple Feature Gradient Descent
Studied the mathematical intuition behind gradient descent and how it works for multiple features. Implemented gradient descent using vector operations, which helps in efficiently updating the parameters in each iteration.


![image](https://github.com/user-attachments/assets/a29384f7-670f-47fa-8613-d51a9fa60d48)


**Implementation Details:**
- Handles n ≥ 2 features
- Simultaneous update of all parameters
- Vectorized computation of partial derivatives
- Batch gradient descent with m training examples

#### Normal Equation: An Alternative Approach
 Learned about the normal equation as an alternative approach to solve linear regression problems without using gradient descent. This method directly computes the optimal parameters.


![image](https://github.com/user-attachments/assets/1091bd4e-196a-48b8-93f0-573519f81b48)


**Key Points:**
- Analytical solution specific to linear regression
- Directly solves for optimal w, b without iterations
- One-shot calculation vs. iterative gradient descent

**Advantages:**
- No need to choose learning rate
- No iteration required
- Works well for smaller feature sets

**Disadvantages:**
- Limited to linear regression only
- Computationally expensive for large feature sets (>10,000 features)
- Doesn't generalize to other learning algorithms

**Important Note:**
- While available in many ML libraries, gradient descent remains the recommended approach
- Understanding both methods helps in choosing the right tool for specific scenarios

#### Key Takeaways 🔑
1. Vectorization dramatically improves computational efficiency
2. NumPy's dot product replaces explicit loops
3. Vector operations enable parallel processing
4. Gradient descent scales elegantly with vectorization
5. Modern hardware (especially GPUs) optimized for vector operations
6. Normal equation provides an alternative analytical solution for linear regression


___
## Day 7
### Topic: Feature Scaling & Gradient Descent Optimization
*Date: October 9, 2024*

### Today's Learning Objectives Completed ✅
- Understanding feature scaling techniques
- Learning to check gradient descent convergence
- Mastering learning rate selection
- Exploring feature engineering concepts
- Introduction to polynomial regression

### Detailed Notes 📝

#### Feature Scaling
Feature scaling is crucial when features have very different ranges of values.

**Why Feature Scaling?**
- Helps gradient descent converge faster
- Makes optimization landscape more symmetric
- Prevents features with larger ranges from dominating

**Common Scaling Methods:**
1. **Simple Scaling (Division by Max)**
   - x₁_scaled = x₁/max(x₁)
   - Example: House size (300-2000 sq ft) → (0.15-1.0)

2. **Mean Normalization**
   - x_normalized = (x - μ)/(max - min)
   - Centers data around zero
   - Range typically: [-1, 1]

3. **Z-score Normalization**
   - x_zscore = (x - μ)/σ
   - μ: mean, σ: standard deviation
   - Typically results in range: [-3, 3]

**When to Scale:**
- When features range from -100 to +100
- When features are very small (e.g., 0.001)
- When features are very large (e.g., house prices in thousands)

#### Checking Gradient Descent Convergence
**Key Methods:**
1. **Plot Learning Curve**
   - X-axis: Number of iterations
   - Y-axis: Cost function J
   - Should see consistent decrease

**Signs of Good Convergence:**
- Cost J decreases after every iteration
- Curve eventually flattens out
- No sudden increases in cost

**Convergence Test:**
- Can use epsilon (ε) threshold (e.g., 0.001)
- If J decreases by less than ε, declare convergence
- Visual inspection often more reliable

#### Choosing Learning Rate (α)
**Guidelines:**
1. Start with small values:
   - Try: 0.001, 0.003, 0.01, 0.03, 0.1
   - Increase by ~3x each time

**Warning Signs:**
- Cost function oscillating → α too large
- Cost increasing consistently → α too large or bug in code
- Very slow decrease → α too small

**Debugging Tip:**
- Try very small α
- If cost still doesn't decrease, check code for bugs

#### Feature Engineering
**Creating New Features:**
- Combine existing features meaningfully
- Transform features to capture relationships
- Use domain knowledge to create relevant features

#### Polynomial Regression
**Extending Linear Regression:**
- Fit non-linear relationships
- Add polynomial terms: x², x³
- Can use different transformations:
  - Square root: √x
  - Powers: x², x³
  - Combinations of features

**Important Considerations:**
- Higher-degree polynomials need more feature scaling
- x² ranges from 1 to 1,000,000 if x ranges from 1 to 1,000
- x³ ranges even larger

### Key Takeaways
1. Feature scaling is crucial for efficient gradient descent
2. Learning curves help diagnose convergence issues
3. Choose learning rate through systematic experimentation
4. Feature engineering can significantly improve model performance
5. Polynomial features allow fitting non-linear relationships

### Personal Notes 📝
Today's learning significantly deepened my understanding of the practical aspects of machine learning optimization. The relationship between feature scaling and gradient descent performance was particularly enlightening. I found the systematic approach to choosing learning rates very practical and will definitely use this in future projects.

___
## Day 8
### Topic: Exploratory Data Analysis with Python
*Date: October 10, 2024*

Today's Learning Objectives Completed ✅
- Initial Data Exploration Techniques
- Data Cleaning and Imputation Methods
- Understanding Relationships in Data
- Practical Applications of EDA

### Detailed Notes 📝
![65d3a069871456bd33730869_GC2xT1oWgAA1rAC](https://github.com/user-attachments/assets/b43524f9-8843-49b7-8dcb-e5ed09046cf6)


#### Initial Data Exploration
- Key functions for first look:
  - `.head()`: Preview first few rows
  - `.info()`: Get overview of data types and missing values
  - `.describe()`: Summary statistics for numerical columns
  - `.value_counts()`: Count categorical values
  - `.dtypes`: Check data types

#### Data Cleaning & Imputation
- Handling Missing Data:
  - Detection using `.isna().sum()`
  - Strategies:
    1. Drop if < 5% missing
    2. Impute with mean/median/mode
    3. Group-based imputation
- Outlier Management:
  - Detection using IQR method
  - Visualization with boxplots
  - Decision points: remove, transform, or keep
  - Impact on distribution and analysis

#### Relationships in Data
- Time-based Analysis:
  - Converting to DateTime using `pd.to_datetime()`
  - Extracting components (year, month, day)
  - Visualizing temporal patterns

- Correlation Analysis:
  - Using `.corr()` for numerical relationships
  - Visualization with heatmaps
  - Understanding correlation strength and direction

- Categorical Relationships:
  - Cross-tabulation with `pd.crosstab()`
  - KDE plots for distribution comparison
  - Categorical variables in scatter plots using hue

#### Practical Applications
- Feature Generation:
  - Creating new columns from existing data
  - Binning numerical data with `pd.cut()`
  - Extracting datetime features

- Hypothesis Generation:
  - Avoiding data snooping/p-hacking
  - Using EDA to form testable hypotheses
  - Understanding limitations of exploratory analysis

#### Key Takeaways 🔑
- EDA is crucial first step in data science workflow
- Balance between cleaning and analysis is important
- Visualization helps identify patterns and relationships
- Always consider statistical significance
- EDA should lead to actionable insights or hypotheses


___
## Day 9
### Topic: Simple Linear Regression in Python & R
*Date: October 11, 2024*

**Today's Learning Objectives Completed ✅**
- Mastered Simple Linear Regression concepts
- Understood Ordinary Least Squares (OLS) method
- Implemented regression in both Python and R
- Visualized and analyzed training/test results

### Detailed Notes 📝

**Simple Linear Regression Fundamentals**
- Linear regression predicts continuous output (y) based on input features (x)
- Model equation: ŷ = b₀ + b₁x
  - b₀: y-intercept (bias)
  - b₁: slope (coefficient)
- Used for predicting numerical values (e.g., salary based on years of experience)

**Ordinary Least Squares (OLS) Method**
- Goal: Minimize sum of squared residuals
- Residual: Difference between actual (yᵢ) and predicted (ŷᵢ) values
- Formula: minimize Σ(yᵢ - ŷᵢ)²
- Finds optimal values for b₀ and b₁ that best fit the data


![image](https://github.com/user-attachments/assets/c3a33019-84b6-4a70-ad0d-8f16fe5f320e)


**Implementation Highlights**

**Python Implementation:**
```python
# Key steps:
1. Data preprocessing
   - Loaded salary data using pandas
   - Split features (X) and target (y)

2. Handling missing values
   - Used SimpleImputer with mean strategy

3. Train-test split
   - 70-30 split ratio
   - Random state set for reproducibility

4. Model training
   - Used sklearn's LinearRegression
   - Fitted on training data

5. Visualization
   - Created scatter plots with seaborn
   - Added regression line for predictions
```
![image](https://github.com/user-attachments/assets/9812b24a-8f55-4640-87a3-80d5d9755cc3)
![image](https://github.com/user-attachments/assets/60bc32fd-bc77-466d-ac6e-df464516cddf)


**R Implementation:**
```r
# Key steps:
1. Data loading and splitting
   - Used caTools for splitting
   - 70-30 ratio maintained

2. Model fitting
   - Used lm() function
   - Formula: Salary ~ YearsExperience

3. Visualization
   - Used ggplot2 for plotting
   - Created separate plots for training and test sets
```
![image](https://github.com/user-attachments/assets/5487e5c4-20ff-409b-9b4e-087a339e9a51)
![image](https://github.com/user-attachments/assets/02696e67-6791-41ec-81c6-92ba9645ac23)
![image](https://github.com/user-attachments/assets/eb6e9fff-12bf-40b0-bb71-aebab8e85da1)

**Key Insights 💡**
- Linear regression works well for this salary prediction case
- The relationship between experience and salary is approximately linear
- Model generalizes well from training to test data
- Both Python and R implementations showed similar results
