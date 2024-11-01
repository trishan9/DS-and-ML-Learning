___
## Day 29
### Topic: Implementation of Decision Trees, Random Forest, and Extra Trees Algorithms
*Date: October 31, 2024*

### Detailed Notes 📝

Today, I implemented Decision Tree, Random Forest, and Extra Trees Classifier algorithms to predict user purchases based on demographic data from the Social Network Ads dataset. The dataset includes features such as age and estimated salary, and the target variable indicates whether the user made a purchase.

#### Code Overview
The project involved several steps, including data loading, preprocessing, model training, evaluation, and visualization. Here's a brief overview of the code structure:

1. **Data Loading and Exploration**:
   The dataset contains records and features related to user demographics and purchase behavior.

2. **Feature Breakdown**:
   - **Demographics**:
     - **Age**: User's age, which ranges from 18 to 60 years.
     - **Estimated Salary**: Annual salary estimates in dollars.
   - **Target Variable**:
     - **Purchased**: Binary variable indicating purchase status (0 for not purchased, 1 for purchased).

3. **Data Preprocessing**:
   The features were separated into `X` (features) and `y` (target variable) for training and testing, and also I applied Feature Scaling after splitting using `StandardScaler`.

4. **Model Training and Evaluation**:
   The dataset was split into training and testing sets, and three classifiers were evaluated: Decision Tree, Random Forest, and Extra Trees. The code for training the models was as follows:
   ```python
   # Decision Tree Classifier
   dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
   dt_classifier.fit(X_train, y_train)

   # Random Forest Classifier
   rf_classifier = RandomForestClassifier(criterion='entropy', random_state=0)
   rf_classifier.fit(X_train, y_train)

   # Extra Trees Classifier
   et_classifier = ExtraTreesClassifier(criterion='entropy', random_state=0)
   et_classifier.fit(X_train, y_train)
   ```

5. **Model Evaluation**:
   The performance of each model was assessed using confusion matrices:
   - **Confusion Matrix and Accuracy for Decision Tree**:
     ```
     [[62  6]
      [ 3 29]]
     ```
     - **Accuracy**: 0.91

   - **Confusion Matrix and Accuracy for Random Forest**:
     ```
     [[63  5]
      [ 4 28]]
     ```
     - **Accuracy**: 0.91

   - **Confusion Matrix and Accuracy for Extra Trees**:
     ```
     [[65  3]
      [ 5 27]]
     ```
     - **Accuracy**: 0.92

#### Results

- **Accuracy**:
  - The final models achieved accuracies of **91%** for the Decision Tree and Random Forest, and **92%** for the Extra Trees Classifier, indicating strong predictive performance across all models.

- **Confusion Matrices**:
  The confusion matrices illustrated the models' predictions, highlighting the number of true positives, true negatives, false positives, and false negatives. The Extra Trees Classifier showed the best performance with the least misclassifications.

![image](https://github.com/user-attachments/assets/12b2383a-9124-407c-a24f-8ee21cf83bec)
![image](https://github.com/user-attachments/assets/55cf3a4b-de24-431a-b1d6-c243b95ee8eb)
![image](https://github.com/user-attachments/assets/3c5d451f-05ab-48df-86c1-19dc0c5e3803)


**I also visualised the decision trees of each model**

For DecisionTreeClassifier:
![tree-1](https://github.com/user-attachments/assets/45787130-e7b1-4a4c-91a7-50e006106702)



#### Reflections
- **Understanding User Purchase Behavior**: This project provided insights into user demographics that influence purchase decisions, reinforcing the importance of data analysis in marketing strategies.

- **Model Evaluation and Selection**: Evaluating multiple classifiers allowed me to appreciate the benefits of ensemble methods, with the Extra Trees Classifier performing slightly better than the other models.

- **Visualization Insights**: Visualizing the decision trees offered a clearer understanding of how decisions are made within each model, contributing to a deeper comprehension of feature importance.
