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
| [Day 10](#day-10)   |         Multiple Linear Regression in Python                                                                                                       |          [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |
| [Day 11](#day-11)   |         Advanced Techniques in Multiple Linear Regression                                                                                                       |          [**Machine Learning A-Z: AI, Python & R**](https://www.udemy.com/course/machinelearning/)                                                                                                                                |
| [Day 12](#day-12)   |         Polynomial Regression                                                                                                       |          [**Machine Learning A-Z**](https://www.udemy.com/course/machinelearning/), [**CampusX**](https://youtu.be/BNWLf3cKdbQ?si=SC-EgkUVHpW2k-Zi)                                                                                                                                |
| [Day 13](#day-13)   |         Bias-Variance Trade-off                                                                                                       |          **StatQuest, CampusX, Krish Naik**                                                                                                                                |
| [Day 14](#day-14)   |         Ridge & Lasso Regression                                                                                                       |          **StatQuest, CampusX, Krish Naik**                                                                                                                                |
| [Day 15](#day-15)   |         Ridge & Lasso Regression Implementation in Python                                                                                                       |          **YouTube**                                                                                                                                |

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


___
## Day 10
### Topic: Multiple Linear Regression in Python
*Date: October 12, 2024*

**Today's Learning Objectives Completed ✅**
- Mastered Multiple Linear Regression in Python with sklearn
- Trained a model to predict startup profits based on multiple features
- Visualized model performance using KDE plots
- Integrated the trained model into a Next.js frontend with a Flask backend into a [VC Profit Predictor](https://vc-profit-predictor.vercel.app/) web application

### Detailed Notes 📝

**Multiple Linear Regression Implementation**

I implemented a multiple linear regression model to predict startup profits based on various features:
- R&D Spend
- Administration Spend
- Marketing Spend
- State (categorical feature)

Key steps in the implementation:

a) Data Preprocessing:
```python
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

b) Handling Missing Values:
```python
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])
```

c) Encoding Categorical Data:
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

d) Train-Test Split:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

e) Model Training:
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

**Model Visualization and Evaluation**

I used KDE plots to visualize the model's performance on unseen data:

```python
sns.set_theme(style="darkgrid")
sns.kdeplot(y_test, color="red", label="Actual Values")
sns.kdeplot(y_hat, color="blue", label="Fitted Values")
plt.title("Actual v/s Fitted Values (Test Set)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/16bc858c-6110-4472-972e-45df936c24a0)

![image](https://github.com/user-attachments/assets/477ad37a-6504-40dd-90cf-ce666ac05d6e)


The KDE plot shows a close alignment between actual and predicted values, indicating good model performance.

**VC Profit Predictor Web Application**

I integrated the trained model into a web application using Next.js for the frontend and Flask for the backend.

**Key features of the application:**
- Input fields for R&D Spend, Administration Spend, Marketing Spend, and State
- Prediction of potential profit based on input values
- Sample data buttons for quick testing
- Clear explanation of the application's purpose and usage

**Screenshots of the application:**

*The main interface of the VC Profit Predictor, showing input fields and prediction result*

![image](https://github.com/user-attachments/assets/e9a17e7c-6568-4091-8aa3-73806e62d940)

*Sample data feature for quick testing of different scenarios*

![image](https://github.com/user-attachments/assets/ec34b64c-967f-4923-93f2-1784741ae975)

**Key Insights 💡**
- Multiple linear regression allows us to consider various factors affecting startup profitability
- The model shows good performance on unseen data, as visualized by the KDE plot
- Integrating ML models into web applications provides an accessible interface for non-technical users
- This tool can help VCs make data-driven investment decisions by analyzing spending patterns and regional variations
- This project demonstrates the practical application of machine learning in a business context, showcasing how data science can inform investment strategies in the venture capital world.


___
## Day 11
### Topic: Advanced Techniques in Multiple Linear Regression
*Date: October 13, 2024*

**Today's Learning Objectives Completed ✅**
- Mastered the concept of dummy variables in regression
- Understood the dummy variable trap and how to avoid it
- Explored statistical significance and p-values in model evaluation
- Learned various model building techniques
- Studied score comparison methods for model evaluation
- Implemented Multiple Linear Regression Model in R
- Completed [VC Profit Predictor](https://github.com/trishan9/VC-Profit-Predictor) Project

### Detailed Notes 📝

### Dummy Variables

- **Definition**: Dummy variables are used to convert categorical data into a numerical format that can be easily used in regression models.
- **Purpose**: They allow the inclusion of categorical variables (like gender, city, or type) in regression models by representing them with numbers.
- **How It Works**:
  - Categorical variables are converted into multiple binary (0 or 1) variables.
  - For a categorical variable with `n` categories, we create `n-1` dummy variables.

![image](https://github.com/user-attachments/assets/ac8724f9-6c5d-4e7d-80c2-4167995f219d)


### Example
Suppose you have a categorical variable called "City" with three categories: Kathmandu, Pokhara, and Lalitpur.
- Create two dummy variables:
  - **Dummy 1**: 1 if Kathmandu, 0 otherwise.
  - **Dummy 2**: 1 if Pokhara, 0 otherwise.
- If both dummy variables are 0, it automatically represents Lalitpur.

### Dummy Variable Trap

- **Definition**: The Dummy Variable Trap occurs when two or more dummy variables are highly correlated (multicollinear), which leads to redundancy in the data.
- **How It Happens**:
  - If you create a dummy variable for each category of a categorical variable (e.g., Kathmandu, Pokhara, Lalitpur), this results in perfect multicollinearity.
  - This means that one dummy variable can be predicted from the others, causing issues in regression models.

![image](https://github.com/user-attachments/assets/a3670aff-2392-473b-b3a8-ee9e870adbea)


#### Example of the Dummy Variable Trap
Imagine you have a regression model with a categorical variable "City" that has three categories: Kathmandu, Pokhara, and Lalitpur.

If you create three dummy variables as follows:
- `Dummy 1` = 1 if Kathmandu, 0 otherwise.
- `Dummy 2` = 1 if Pokhara, 0 otherwise.
- `Dummy 3` = 1 if Lalitpur, 0 otherwise.

In this setup, there's a problem:
- If Dummy 1 and Dummy 2 are both 0, then we know for sure that Dummy 3 must be 1.
- This creates a linear relationship between the dummy variables (`Dummy 3 = 1 - Dummy 1 - Dummy 2`), leading to multicollinearity.

#### Solution to Avoid the Dummy Variable Trap
- **Always use `n-1` dummy variables for `n` categories.**
  - For the example above, we should use only two dummy variables instead of three.
  - Possible setup:
    - `Dummy 1` = 1 if Kathmandu, 0 otherwise.
    - `Dummy 2` = 1 if Pokhara, 0 otherwise.
    - The absence of both (i.e., 0 for both Dummy 1 and Dummy 2) would mean the observation is for Lalitpur.

### Intuition Behind the Dummy Variable Trap
- The trap happens because including all the dummy variables introduces redundancy.
- It leads to issues in interpreting the regression coefficients and reduces the statistical significance of predictors.
- By avoiding the trap, you ensure that the model runs properly and gives more reliable results.

### Statistical Significance Level

- **Definition**: A significance level is a threshold that determines how confident we are in rejecting the null hypothesis for a predictor in the regression model.
- **Common Levels**: Commonly used levels include 5% (0.05), 1% (0.01), and 10% (0.10).
  - A 5% significance level means we are 95% confident that the predictor's effect on the target variable is not due to chance.

### P-Value

- **Definition**: The p-value measures the probability that the observed relationship between the predictor and the target variable occurred by random chance.
- **Interpretation**:
  - If **p-value < significance level**: Reject the null hypothesis, indicating that the predictor is statistically significant.
  - If **p-value >= significance level**: Fail to reject the null hypothesis, meaning that the predictor does not have a significant effect on the target variable.

### Intuition Behind Statistical Significance
- It helps to decide whether a predictor variable genuinely impacts the target variable or if the observed effect could have occurred by chance.
- A lower p-value indicates that the predictor is more likely to have a meaningful contribution to the model.

### Building a Model

Building an effective regression model involves selecting the right predictor variables. There are several common approaches to model building:
### 1. All-in Approach

- **Definition**: This approach involves including all available predictors in the regression model regardless of their significance.
- **Use Cases**:
  - Useful when you have prior knowledge or domain expertise that indicates all variables play an essential role.
  - When you're conducting exploratory data analysis and want to assess the collective effect of all predictors.
- **Drawback**: Including irrelevant predictors can lead to overfitting, where the model fits the training data too well and performs poorly on new data.

### 2. Backward Elimination

- **Process**:
  1. Start with all the predictors in the model.
  2. Identify the predictor with the highest p-value greater than the chosen significance level.
  3. Remove this predictor from the model.
  4. Refit the model and repeat the process until all remaining predictors are statistically significant.
- **Benefit**: It systematically reduces model complexity by eliminating less relevant variables.
- **Intuition**: Focuses on simplifying the model by removing predictors that do not provide meaningful information.

![image](https://github.com/user-attachments/assets/7f22ce70-bccc-496a-81d3-99dec934ae9b)


### 3. Forward Selection

- **Process**:
  1. Start with no predictors in the model.
  2. Add the predictor with the lowest p-value that improves the model's performance.
  3. Continue adding predictors one by one until no significant improvement is observed.
- **Use Case**: Ideal when you have a large number of predictors and want to build a simple model starting from scratch.
- **Drawback**: Forward selection might miss some combinations of predictors that could become significant when considered together.

### 4. Bidirectional Elimination

- **Definition**: Combines Forward Selection and Backward Elimination in one method.
- **Process**: At each step, it adds significant predictors and removes those that become non-significant.
- **Benefit**: It evaluates the significance of variables as you add and remove them, providing a balanced approach to model building.
- **Intuition**: Ensures that as new variables are introduced, existing variables are still relevant to the model.

### Score Comparison
To evaluate and compare different regression models, the following metrics are most commonly used:

### 1. R-squared (Coefficient of Determination)

- **Definition**: Measures the proportion of variance in the target variable that can be explained by the predictor variables.
- **Range**: 0 to 1 (or 0% to 100%).
  - **Closer to 1** indicates that a larger portion of the variance is explained by the model.
  - **Closer to 0** suggests that the model does not explain much of the variability in the data.
- **Limitations**: Adding more predictors always increases R-squared, even if those predictors are not relevant.

### 2. Adjusted R-squared

- **Definition**: Adjusted R-squared modifies the R-squared value to account for the number of predictors in the model.
- **Interpretation**:
  - It only increases if the newly added predictor improves the model more than would be expected by chance.
  - If a predictor does not improve the model, Adjusted R-squared decreases.
- **Use Case**: It helps to avoid overfitting by ensuring that each added predictor genuinely contributes to improving the model.

### Intuition for Score Comparison

- **R-squared** tells you how well the model fits the data overall.
- **Adjusted R-squared** prevents you from adding irrelevant predictors by adjusting for model complexity.
- Use these metrics to select the best model that balances simplicity and predictive power.

### Multiple Linear Regression Implementation in R
![image](https://github.com/user-attachments/assets/ebcdc439-2382-4223-8ece-92b4b69e5123)

### VC Profit Predictor Project Completion
https://github.com/user-attachments/assets/06c969d9-e93a-4e24-bc23-8d50282cb001

### Key Insights 💡
- Proper handling of categorical variables is crucial for accurate regression models
- Statistical significance helps identify truly impactful predictors
- Different model building techniques suit various scenarios and data characteristics
- Balancing model complexity and predictive power is key in regression analysis


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


___
## Day 13
### Topic: Bias-Variance Trade-off
*Date: October 15, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding the concept of Bias-Variance Trade-off
- Exploring key components: Bias and Variance
- Analyzing the effects of high bias and high variance
- Learning strategies to manage the trade-off
- Studying examples in different ML models

### Detailed Notes 📝

### Introduction to Bias-Variance Trade-off

![image](https://github.com/user-attachments/assets/c2b2e215-cf26-45e4-900b-3e2c293e0ff5)

- Fundamental concept in machine learning
- Describes the balance between two sources of errors: bias and variance
- Crucial for building models that generalize well to new, unseen data

### Key Concepts

#### Bias
- inability of ML model to truly capture relationships in training data.
- Bias refers to the error introduced by the simplifying assumptions a model makes to learn from the training data.
- High bias means that the model is too simple and makes assumptions that may not align well with the data, leading to **underfitting**.
- A model with high bias pays little attention to the training data, oversimplifies the problem, and misses relevant relationships between the features and the target outputs.

#### Variance
- difference of the fits between different datasets.
- Variance refers to the error introduced by the model's sensitivity to the small fluctuations in the training data.
- High variance means that the model is too complex and captures the noise in the training data along with the actual data patterns, leading to **overfitting**.
- A model with high variance pays too much attention to the training data, learns its details and noise, and fails to generalize well to new data.

### The Trade-off

- The goal in machine learning is to develop a model that has both low bias and low variance, but in practice, reducing one often increases the other.
- The **Bias-Variance Trade-off** refers to the challenge of finding the right balance between bias and variance to minimize the **total error**.

### Effects of High Bias and High Variance

![image](https://github.com/user-attachments/assets/72bd0c8e-4b73-4439-9bc5-2754e0c5b6c5)

#### High Bias (Underfitting)

![image](https://github.com/user-attachments/assets/b8b77e16-5eb0-41c4-97c8-8341fe19050b)

- Model is too simple.
- Has low complexity.
- Does not capture the underlying patterns in the data.
- Leads to high training error and high testing error.
- Examples: Linear Regression on a nonlinear dataset, or a shallow decision tree.

#### High Variance (Overfitting)

![image](https://github.com/user-attachments/assets/8fcf10ec-6c34-4d54-a828-98362d8e9960)

- training error is extremely low but test error is high.
- Model is too complex.
- Has high flexibility.
- Captures noise or irrelevant patterns in the training data.
- Leads to low training error but high testing error.
- Examples: Deep decision trees, k-NN with very low values of k.

### Finding the Optimal Balance

![image](https://github.com/user-attachments/assets/3d1a705c-a0dc-4ad7-aacd-8a23f50f3ef6)

To achieve the best performance, a model should have:
- Low enough bias to make predictions that are on average correct.
- Low enough variance to generalize well to new data.

**Steps to Find the Balance:**
1. **Start Simple and Increase Complexity:** Begin with a simple model and increase complexity gradually to observe how the model performs on training and validation sets.
2. **Use Cross-Validation:** Split your data into multiple training and testing sets to better understand how well the model generalizes.
3. **Regularization Techniques:** Use techniques like Lasso, Ridge, or Elastic Net to prevent overfitting by penalizing large coefficients in the model.
4. **Feature Engineering:** Select the most relevant features to reduce complexity and noise in the data.
5. **Ensemble Methods:** Use methods like Bagging, Boosting, or Stacking to combine predictions from multiple models to reduce bias and variance.

### Examples in ML Models

1. **Linear Regression:**
   - High Bias: Assumes a linear relationship, which may not fit nonlinear data well.
   - Low Variance: Small changes in the data do not drastically change the predictions.

2. **Polynomial Regression:**
   - Low Bias: Can fit complex patterns in the data.
   - High Variance: Overfits the data if the polynomial degree is too high.

### Strategies to Manage Bias-Variance Trade-off

1. **Adjust Model Complexity:** Increase complexity to reduce bias or decrease complexity to reduce variance.
2. **Regularization Techniques:**
   - L1 Regularization (Lasso): Shrinks coefficients, reducing variance.
   - L2 Regularization (Ridge): Distributes error among features, reducing variance.
3. **Use More Training Data:** Helps to reduce variance without affecting bias significantly.
4. **Ensemble Methods:**
   - **Bagging:** Reduces variance by averaging multiple models (e.g., Random Forest).
   - **Boosting:** Reduces bias by sequentially building models that correct the errors of the previous ones.

### Key Insights 🔑
- Bias leads to underfitting, variance leads to overfitting
- The goal is to find a balance between simplicity and complexity
- Techniques like cross-validation, regularization, and ensemble methods help manage the trade-off
- Understanding this concept is crucial for building robust, generalizable ML models
- Different models have different bias-variance characteristics
- The optimal model minimizes both training and validation errors


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
