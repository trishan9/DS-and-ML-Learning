___
## Day 17
### Topic: Decision Tree Regression
*Date: October 19, 2024*

**Today's Learning Objectives Completed ✅**
- Understanding the concept and mechanics of Decision Tree Regression
- Exploring the process of building a decision tree for regression tasks
- Learning about splitting criteria and prediction methods in decision trees
- Identifying use cases, advantages, and limitations of Decision Tree Regression

### Detailed Notes 📝

### Decision Tree Regression

**Decision Tree Regression** is a machine learning algorithm used to predict continuous values, making it a popular method for regression tasks. It works by creating a tree-like structure of decision rules that split the data into different subsets based on the values of the input features. Let's break down how Decision Tree Regression works, its components, and how it makes predictions.

#### What is Decision Tree Regression?

Decision Tree Regression is a type of regression model that predicts a target value by learning simple decision rules inferred from the data's features. It builds a tree structure where:
- **Nodes** represent conditions based on feature values.
- **Branches** represent the outcomes of those conditions.
- **Leaf nodes** contain the predicted values.

The goal of the tree is to partition the dataset into subsets that have similar target values.

#### How Does It Work?

The process of building a decision tree can be summarized in these steps:
1. **Choose the Best Feature for Splitting:** The algorithm examines all features and different values of those features to determine the best way to split the data. The split is chosen based on a criterion that minimizes the prediction error (often using **Mean Squared Error (MSE)**).

2. **Create Branches Based on Conditions:** It creates branches based on the chosen split, which divides the dataset into more homogenous groups with similar target values.

3. **Repeat the Process:** The splitting continues recursively on each branch, creating a deeper tree structure, until a stopping condition is reached (like a maximum tree depth or a minimum number of samples in a node).

4. **Assign Values at Leaf Nodes:** Once the data cannot be split further, the algorithm assigns the **mean** of the target values in each leaf node as the prediction.

#### Making Predictions

When a new data point is fed into the tree, it travels through the nodes based on the decision conditions until it reaches a leaf node. The prediction for that data point is the **mean** of the target values of the training data that fell into that leaf node.

**Example:**
Consider a dataset for predicting the price of Mo:Mo in different cities of Nepal based on size and location:
- If a decision tree splits the data at **location = Kathmandu**, it creates two branches:
  - **Left branch** for Mo:Mo prices in Kathmandu
  - **Right branch** for Mo:Mo prices outside Kathmandu (e.g., Pokhara, Biratnagar)
- If a new data point corresponds to Mo:Mo in Kathmandu, it will go to the left branch, where the predicted price might be the mean price of Mo:Mo in similar locations within Kathmandu.

#### How are the Conditions Set?

The conditions for splitting nodes are set to minimize the prediction error, typically using **Mean Squared Error (MSE)**. The algorithm tries different values of each feature and chooses the split that results in the greatest reduction in error.

![image](https://github.com/user-attachments/assets/3f240240-856a-40d2-b9c4-16da6a2dcd1c)


#### How is the Predicted Value Set?

The predicted value at each leaf node is usually the **mean** of the target values of all the data points in that node. This means that all samples that fall into the same leaf node will have the same prediction.

![image](https://github.com/user-attachments/assets/224418dc-b37e-42f7-b52e-5d043040c5f2)
![image](https://github.com/user-attachments/assets/7bcfbde9-5e3b-4c92-a352-9db04bd2255a)

**Example:**
If a leaf node contains the prices of Mo:Mo in Kathmandu as Rs. 150, Rs. 160, and Rs. 170, the predicted price for any new Mo:Mo order reaching this node will be:
Predicted value = (150 + 160 + 170) / 3 = Rs. 160

#### Use Cases of Decision Tree Regression

Decision Tree Regression can be used in various real-world scenarios:
- **Predicting House Prices:** Based on features like the number of rooms, size, location, etc.
- **Stock Market Analysis:** To forecast stock prices based on historical data.
- **Weather Prediction:** Estimating temperatures or rainfall based on previous patterns.

#### Advantages

- **Easy to Understand and Interpret:** The tree structure is simple to visualize, making it easy to explain the decision-making process.
- **Non-Linear Relationships:** Can handle complex data relationships and patterns without requiring feature scaling or transformation.
- **Feature Importance:** It highlights which features are most important in making predictions.

#### Limitations

- **Overfitting:** Decision trees can easily overfit the training data if they grow too deep, capturing noise instead of patterns.
- **Piecewise Constant Prediction:** The predictions are not smooth but have a step-like structure because all points in the same leaf node have the same predicted value.
- **Sensitive to Small Changes:** Small changes in the data can lead to different splits, resulting in a completely different tree.

#### How Decision Tree Regression Handles Predictions

- When splitting the data, the goal is to create groups where the target values are as similar as possible.
- The predicted value at the leaf node is calculated using the mean of the target values, as it minimizes the squared error.
- Therefore, **any data point that falls into the same leaf node will get the same predicted value**, which is the mean of that group.

#### Summary of Decision Tree Regression

- **Split Criteria:** The algorithm chooses splits that minimize the prediction error using MSE.
- **Predicted Value:** At each leaf node, the mean of the target values is used to make predictions.
- **Advantages:** Easy to interpret, handles non-linear relationships, and shows feature importance.
- **Limitations:** Prone to overfitting, creates piecewise constant predictions, and is sensitive to data variations.

#### Intuition and Visualization

Like a flowchart where each question narrows down the possibilities until you reach a final answer. The splits in the tree are like a series of yes/no questions that guide you toward a prediction. By using averages in the leaf nodes, Decision Tree Regression provides the best guess based on similar cases.

This makes Decision Tree Regression a powerful and intuitive tool for making predictions, especially when dealing with non-linear data patterns.

### Key Takeaways 🔑
- Decision trees split data recursively to create homogeneous groups
- Predictions are based on the mean value of samples in leaf nodes
- Easy to interpret but can overfit if not properly tuned
- Useful for capturing non-linear patterns in data
- Provides insights into feature importance
