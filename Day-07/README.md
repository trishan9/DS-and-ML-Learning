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
