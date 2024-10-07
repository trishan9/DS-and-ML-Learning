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
