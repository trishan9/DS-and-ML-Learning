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
