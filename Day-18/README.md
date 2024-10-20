___
## Day 18
### Topic: Implementing Decision Tree Regression in Python
*Date: October 20, 2024*

**Today's Learning Objectives Completed ✅**
- Implemented Decision Tree Regression on two different datasets
- Explored the impact of single vs. multi-feature datasets on model performance
- Practiced data loading, preparation, and model evaluation techniques
- Visualized Decision Tree Regression predictions and errors

### Detailed Notes 📝

Today, I implemented a Decision Tree Regression model using Python. This technique is useful for predicting continuous values based on the given features. I worked with two datasets: one for a salary prediction based on position levels and another for housing prices in California.

### Steps Taken

1. **Data Loading**
   - I started by loading the dataset using pandas. The first dataset contained positions, levels, and corresponding salaries.
   - For the second dataset, I fetched California housing data.

2. **Data Preparation**
   - I extracted the features (`X`) and target values (`y`) from the datasets.
   - For the salary dataset, I selected the levels as features and salaries as target values.

3. **Model Implementation**
   - I imported the `DecisionTreeRegressor` from `sklearn`.
   - I created an instance of the model and fitted it to the salary dataset.

4. **Making Predictions**
   - After training, I predicted the salary for a position level of 10.
   - To visualize the results, I plotted the actual salaries against the predicted values using matplotlib.

5. **Model Evaluation**
   - I evaluated the model using the R² score and Mean Squared Error (MSE).
   - The results showed an R² score of 1.0 and an MSE of 0.0, indicating perfect predictions on the training dataset.

   ```python
   print(f"R2 Score: {r2_score(y, regressor.predict(X))}")
   print(f"MSE: {mean_squared_error(y, regressor.predict(X))}")
   ```

6. **Second Dataset Implementation**
   - I then moved on to the California housing dataset.
   - I split the data into training and testing sets using `train_test_split`.

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

7. **Training the Model**
   - I trained the Decision Tree Regressor on the training set and made predictions on the testing set.

8. **Evaluation of Predictions**
   - I visualized the differences between predicted and actual values using seaborn's KDE plot.
   - I calculated the R² score and MSE for the predictions, which resulted in an R² score of approximately 0.57 and an MSE of about 0.54.


### Conclusion
- Implementing Decision Tree Regression allowed me to understand how this model can effectively predict continuous values from different datasets. The perfect results on the salary dataset indicate that the model captured the underlying patterns well, while the California housing data showed a more realistic performance. I found the visualizations helpful for assessing the model's accuracy and making sense of the predictions.
- Decision Tree Regression performs better with larger feature sets
- Multi-feature datasets provide more information for the model to learn from
- Single feature datasets may lead to overfitting or simplistic models

### Key Observations

- Single Feature Dataset (Position Salaries):
  - Perfect R² score (1.0) and MSE (0.0) on training data
  - Indicates potential overfitting to the training data

  ![image](https://github.com/user-attachments/assets/860d18b1-d204-4929-af3f-2fda9ea76ec0)


- Multi-feature Dataset (California Housing):
  - R² score: ~0.57
  - MSE: ~0.54
  - More realistic performance, showing the complexity of the problem

![image](https://github.com/user-attachments/assets/71b5db93-2004-4797-bb00-17df9a65e8fa)


### Key Takeaways 🔑

- Decision trees can capture complex patterns in multi-feature datasets
- Perfect scores on training data may indicate overfitting
- Visualization helps in understanding model performance and errors
- R² score and MSE provide quantitative measures of model accuracy
- Decision Tree Regression is more suitable for datasets with multiple features
