___
## Day 27
### Topic: Implementation of Classification Algorithms for Student Performance Evaluation
*Date: October 29, 2024*

### Detailed Notes 📝

In today's project, I focused on evaluating student performance using various metrics gathered from a dataset. The main objective was to analyze the factors affecting students' academic performance and to build predictive models to classify their grade classes based on their GPA.

**Dataset Overview:**
The dataset consists of **2392 records** with **15 features** that capture different aspects of student performance. Here are the key features included:

- **Demographics:**
  - **Age:** Ranges from 15 to 18 years.
  - **Gender:** Coded as 0 (Male) and 1 (Female).
  - **Ethnicity:** Coded into categories such as Caucasian, African American, Asian, and Other.
  - **Parental Education:** Represents the education level of parents from None to Higher.

- **Study Habits:**
  - **StudyTimeWeekly:** The number of hours students study weekly (0 to 20 hours).
  - **Absences:** Total number of absences during the school year (0 to 30).
  - **Tutoring:** Indicates whether a student has received tutoring (0: No, 1: Yes).

- **Parental Involvement and Extracurricular Activities:**
  - **Parental Support:** Levels of support provided by parents, coded from None to Very High.
  - Participation in extracurricular activities (e.g., sports, music, volunteering).

- **Academic Performance:**
  - **GPA:** Grade Point Average ranging from 2.0 to 4.0.
  - **Grade Class:** Target variable representing classification based on GPA:
    - 0: 'A' (GPA >= 3.5)
    - 1: 'B' (3.0 <= GPA < 3.5)
    - 2: 'C' (2.5 <= GPA < 3.0)
    - 3: 'D' (2.0 <= GPA < 2.5)
    - 4: 'F' (GPA < 2.0)

---

**Exploratory Data Analysis (EDA):**
I performed EDA to understand the distribution of grades and the relationships between various factors and academic performance:

1. **Grade Distribution:** A pie chart displayed the distribution of students across different grade classes.
2. **Boxplots:**
   - **GPA by Grade Class:** Analyzed how GPA varies with different grade classifications.
   - **Absences by Grade Class:** Investigated the relationship between absences and grade class.
   - **Study Time by Grade Class:** Explored the correlation between weekly study hours and academic performance.

---

**Model Building:**
Using various classification models, I aimed to predict student performance based on the features available:

1. **Data Preparation:**
   - Split the dataset into training and testing sets (70/30 split).
   - Standardized the features using `StandardScaler`.

2. **Model Selection:**
   - Evaluated several models including Decision Trees, Random Forests, SVMs, KNN, and Logistic Regression using **GridSearchCV** for hyperparameter tuning.
   - Implemented **K-Fold Cross Validation** to ensure model robustness.

3. **Results:**
   - The best model was selected based on accuracy scores from cross-validation.
   - The final model achieved an accuracy of **94.85%** on the test set, with a comprehensive classification report showcasing precision, recall, and F1 scores across different grade classes.

4. **Confusion Matrix:**
   - Displayed the confusion matrix for the best-fitted model, providing insights into the model's performance in classifying each grade.

---

**Results**

- **Accuracy**:
  - The final model achieved an accuracy of around **94.8%**, indicating strong predictive performance.

- **Classification Report**:
  The classification report displayed precision, recall, and F1 scores for each class:
  ```
                precision    recall  f1-score   support

          0.0       0.78      0.84      0.81        43
          1.0       0.88      0.89      0.88        72
          2.0       0.98      0.95      0.97       107
          3.0       0.91      0.96      0.93       112
          4.0       0.99      0.97      0.98       384

      accuracy                           0.95       718
     macro avg       0.91      0.92      0.91       718
  weighted avg       0.95      0.95      0.95       718
  ```
  This indicates balanced performance across classes, with high precision and recall for most grade classes, particularly for lower grades (2.0 and 3.0).

- **Confusion Matrix**:
  The confusion matrix illustrated the model's predictions, showing true positives, true negatives, false positives, and false negatives. It provided insights into areas of misclassification, particularly in lower-performing grade classes.

![image](https://github.com/user-attachments/assets/f209805d-7daf-45a4-b0af-3c81ba283f9d)
![image](https://github.com/user-attachments/assets/0bbd3da4-8faa-474d-8c7a-58d0119ee238)
![image](https://github.com/user-attachments/assets/75b99575-75ea-418d-b3df-2e6caee9a1de)
![image](https://github.com/user-attachments/assets/4da2dbcb-4eb7-49da-a16b-b92bc7c5cfb1)

---

**Reflections:**

- **Understanding Student Performance Factors**: This project allowed me to explore the multifaceted factors influencing student performance, reinforcing the importance of demographic and behavioral attributes in academic outcomes.

- **Model Evaluation and Selection**: Through evaluating multiple models, I gained insights into the significance of hyperparameter tuning and model selection based on data characteristics. The Decision Tree emerged as the most effective model for this dataset.

- **Data Visualization Insights**: Visualizing the data revealed critical patterns, such as the relationship between study time and grade class, which can inform educational strategies and interventions.

- **Practical Implications**: The findings from this project have practical applications in educational settings, enabling educators to identify at-risk students and develop targeted support strategies based on empirical data.

- **Future Enhancements**: Opportunities for further improvement include integrating more features, such as behavioral metrics or historical performance, and exploring advanced techniques like ensemble methods or deep learning models to enhance predictive capabilities.

Overall, this project was an enriching experience that deepened my understanding of data analysis, model evaluation, and the practical implications of different classification algorithms.
