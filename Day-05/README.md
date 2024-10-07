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
