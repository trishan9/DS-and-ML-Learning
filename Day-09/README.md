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
