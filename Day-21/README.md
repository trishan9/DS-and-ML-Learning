___
## Day 21
### Topic: Cost Function and Gradient Descent in Logistic Regression
*Date: October 23, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding Cost Function for Logistic Regression
- Log Loss Cost Function
- Gradient Descent Optimization for choosing best params for Logistic Regression Model

### Detailed Notes 📝

When training a logistic regression model, we need a way to evaluate how well the model's parameters (weights, \( w \), and bias, \( b \)) fit the training data. This is where the **cost function** comes in. The cost function provides a way to measure the error of the model's predictions and helps us adjust the model's parameters to improve its performance.

### Why Not Use Squared Error Cost Function for Logistic Regression?

In linear regression, we typically use the **squared error cost function**, which works well because the relationship between the features and the target is linear. However, in **logistic regression**, which is used for **binary classification** (where the target variable \( y \) can only be 0 or 1), this cost function doesn't work well. Here's why:

- Logistic regression predicts probabilities that lie between 0 and 1, using the **sigmoid function** \( f(x) = \frac{1}{1 + e^{-(wx + b)}} \).
- If we apply the squared error cost function, the resulting **cost surface** (a plot of the cost values for different parameters) is **non-convex**. This means the cost surface has many local minima (dips), which makes it difficult for **gradient descent** to find the **global minimum**.

### The Logistic Regression Cost Function

![image](https://github.com/user-attachments/assets/fe2b6560-a2e7-4274-beba-e67c8998a4ba)


To ensure a smooth, convex cost surface (which guarantees convergence to the global minimum), we use a different cost function for logistic regression. Here's how it works:

1. **Loss Function for a Single Example**: We define the **loss function** to measure how well the model predicts a single training example. The loss depends on whether the true label \( y \) is 1 or 0:
   - If \( y = 1 \): the loss is -log(f(x)) where f(x) is the predicted probability.
   - If \( y = 0 \): the loss is -log(1 - f(x)), where 1 - f(x) is the predicted probability of the label being 0.

   The idea is that if the model's prediction is **close to the true label**, the loss will be small. But if the prediction is far from the true label, the loss will be large. For example:
   - If \( y = 1 \) and the model predicts a probability close to 1, the loss is close to 0.
   - If \( y = 1 \) but the model predicts a probability close to 0, the loss becomes very large.

   Similarly:
   - If \( y = 0 \) and the model predicts a probability close to 0, the loss is close to 0.
   - If \( y = 0 \) but the model predicts a probability close to 1, the loss becomes large.

2. **Cost Function for the Entire Training Set**: The **cost function** aggregates the losses over all training examples to measure how well the model is doing overall:
   \
   J(w, b) = 1/m ∑ L(f(x), y)
   \

   This cost function is **convex**, meaning that gradient descent can reliably find the global minimum, ensuring that we find the best parameters.

3. **Simplified Cost Function**:
4.
![image](https://github.com/user-attachments/assets/cffd2b97-efb1-41bf-bcbd-5847f86152e9)


### Gradient Descent for Logistic Regression

Now that we have a proper cost function, the goal is to find the parameters \( w \) and \( b \) that minimize the cost function, i.e., the parameters that make the model's predictions as accurate as possible.

**Gradient Descent** is the algorithm used for this. Here's how it works:

1. Parameter Initialization:
   - Start with random or zero values for w and b

2. Iterative Updates:
   ```
   w := w - α ∂J(w,b)/∂w
   b := b - α ∂J(w,b)/∂b
   ```
   where α = learning rate

3. Convergence:
   - Continue updates until cost function stabilizes
   - Parameters reach optimal values for classification

![image](https://github.com/user-attachments/assets/e8104542-8997-43ab-9fd4-6a4e7b2433bf)

### Optimization Characteristics
- Cost function is convex (bowl-shaped)
- Guaranteed to find global minimum
- Learning rate α controls step size:
  - Too large: May overshoot
  - Too small: Slow convergence

### Summary

- **Cost function**: Measures how well the model's parameters fit the data. For logistic regression, we use a specific cost function derived from the log-loss, ensuring a smooth, convex surface.
- **Gradient descent**: An iterative algorithm that adjusts the model's parameters by following the negative gradient of the cost function until it finds the parameters that minimize the cost.

This process helps the logistic regression model learn the best values for \( w \) and \( b \), so it can make accurate predictions on new data.

### Key Takeaways 🔑
- Logistic regression needs special cost function (log loss)
- Cost function measures prediction accuracy
- Gradient descent finds optimal parameters
- Convex surface ensures convergence
- Learning rate crucial for optimization
- Process iteratively improves model performance
