___
## Day 25
### Topic: Implementation of Support Vector Machines
*Date: October 27, 2024*

### Detailed Notes 📝

---

Today, I successfully implemented a Support Vector Machine (SVM) classification model using the `sklearn` library to perform sentiment analysis on a tweets dataset. The dataset initially contained over 70,000 rows, but to prevent my laptop from hanging during model fitting, I limited the data to the first 10,000 entries.

This project implements a sentiment analysis system using Support Vector Machine (SVM) classification on Twitter data. The system classifies the given text into four categories: Positive, Negative, Neutral, and Irrelevant. It demonstrates the practical application of machine learning in natural language processing (NLP).

### **Steps Taken:**

1. **Data Loading and Preprocessing:**
   - Loaded the dataset using `pandas` and displayed the first few entries.
   - Renamed the columns for clarity, setting them as `ID`, `Category`, `Sentiment`, and `Text`.
   - Dropped any rows with missing text values to ensure a clean dataset.

2. **Feature and Label Extraction:**
   - Extracted the text data (`X`) and the corresponding sentiment labels (`y`) for model training and evaluation.

3. **Data Splitting:**
   - Utilized `train_test_split` to divide the dataset into training and testing sets, allocating 20% for testing.

4. **Label Encoding:**
   - Implemented `LabelEncoder` to convert sentiment labels into a numerical format suitable for classification.

5. **Text Vectorization and Model Pipeline:**
   - Created a pipeline with `TfidfVectorizer` for text feature extraction (Text Vectorization) and `SVC` with a linear kernel for classification.
   - Fitted the model to the training data.

6. **Model Testing:**
   - Conducted predictions on sample texts to verify the model's functionality:
     - "Hello what's up buddie?" → **Irrelevant**
     - "I am so happy today!" → **Positive**
     - "I am pissed off!" → **Negative**

7. **Model Evaluation:**
   - Predicted sentiment labels on the test dataset and calculated the accuracy, which was approximately **92.67%**.
   - Generated a classification report, detailing precision, recall, and F1 scores for each sentiment category.

### **Results:**
- **Accuracy:** 0.9267
- **Classification Report:**
  - Precision, Recall, F1-Score, and Support for each sentiment category were provided, showing balanced performance across classes.

![image](https://github.com/user-attachments/assets/3ac7afab-6827-4155-a2a4-7a17e7fa18e9)
![image](https://github.com/user-attachments/assets/0d109468-e5ea-4ef2-a152-48c6cd86b63d)
![image](https://github.com/user-attachments/assets/044e4581-44bc-4c0c-95e9-76680d47a280)

---

### **Reflections:**
Implementing SVM was an enriching experience, especially after diving deep into its intuition and mechanics. The effectiveness of SVM in handling classification tasks makes it one of the best classifiers available, particularly when dealing with high-dimensional data like text.

- **Understanding SVM:** I found the theoretical concepts surrounding SVM—such as the decision boundary, support vectors, and the margin—to be intellectually stimulating. This deep dive into the algorithm's workings has significantly enhanced my understanding of machine learning.
- **Practical Application:** Applying the theoretical knowledge practically solidified my learning and highlighted the importance of preprocessing, feature extraction, and model evaluation in building effective machine learning solutions.
- **Model Performance:** The impressive accuracy and detailed classification report provided confidence in the model's ability to generalize. It was rewarding to see how minor adjustments in data handling and model configuration led to substantial improvements in performance.
- **Interest in NLP:** This project ignited my enthusiasm for natural language processing (NLP). The ability to extract sentiment from tweets is not only fascinating but also has practical implications in fields such as marketing, social media analysis, and public opinion research.

---
