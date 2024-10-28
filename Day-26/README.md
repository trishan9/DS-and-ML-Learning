___
## Day 26
### Topic: Implementation of Image Classification for Brain Tumor Detection
*Date: October 28, 2024*

### Detailed Notes 📝

Today, I implemented an image classification model aimed at detecting brain tumors using a dataset of MRI-scanned images. The project involved data preprocessing, training various classifiers, and evaluating their performance, ultimately leading to the selection of Support Vector Machine (SVM) with an `rbf` kernel due to its superior accuracy in cross-validation tests.

**Steps Taken:**

1. **Data Loading and Exploration:**
   - Loaded the dataset containing file names of MRI images and their corresponding classes (0 for non-tumor and 1 for tumor) using `pandas`.
   - Displayed the first few entries of the dataset to confirm its structure and contents.

**[Dataset](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/) Overview**:

![image](https://github.com/user-attachments/assets/abe304c7-7794-47b9-9a1b-79af2a9046e4)


2. **Image Preprocessing:**
   - Defined a function `preprocess_image` to read and preprocess images:
     - Read images as grayscale for consistency.
     - Resized each image to 28x28 pixels, flattening the array for input into the model.
   - Applied this function to the dataset, generating feature vectors (`X`) from the image files.

3. **Data Splitting:**
   - Used `train_test_split` to divide the dataset into training and testing sets, allocating 20% for testing. This ensured that the model could be evaluated on unseen data, providing insights into its performance.

4. **Model Selection and Training:**
   - Prepared to evaluate multiple classification models, including **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **SVM** with an `rbf` kernel (as, till now, I have only learnt these 3 classification algorithms).
   - Implemented 10-fold cross-validation using to assess the models’ performance reliably.

5. **Cross-Validation and Visualization:**
   - Trained each model pipeline (including standard scaling and the classifier) and recorded cross-validation scores.
   - Created a box plot to visualize the performance of each model, revealing that SVM outperformed the others.

6. **Final Model Training:**
   - Based on the cross-validation results, I selected the SVM model with an `rbf` kernel and trained it using the training dataset.

7. **Prediction on Test Images:**
   - Tested the trained model on two sample MRI images:
     - An image with a brain tumor.
     - A non-tumor image.
   - Displayed the images and printed the prediction results, confirming the model's ability to classify the images accurately.

8. **Model Evaluation:**
   - Made predictions on the test set and evaluated the model using accuracy, classification report, and confusion matrix metrics.
   - The accuracy was around **93.3%**, indicating the model's effectiveness in distinguishing between tumor and non-tumor images.

---

**Results:**
- **Accuracy:** 0.9336
- **Classification Report:**
  - The report highlighted the precision, recall, and F1 scores for each class, indicating balanced performance:
    - Non-Tumor (0): Precision of 0.93, Recall of 0.96
    - Tumor (1): Precision of 0.94, Recall of 0.90
  - The overall metrics showed the model's reliability in both classes.

- **Confusion Matrix:**
  - Confusion Matrix displayed the true positives, true negatives, false positives, and false negatives:
    ```
    [[412  19]
     [ 31 291]]
    ```
  - The matrix suggested that while the model performed well, there were some misclassifications, particularly in the false negatives (acceptable, I guess, as we are using SVM that utilizes soft margins for misclassifications).

![image](https://github.com/user-attachments/assets/431cd1f9-edad-49a2-8b4e-aedb9a00bda5)
![image](https://github.com/user-attachments/assets/2ceb4701-f17a-438a-bacb-7366e4d5e59f)
![image](https://github.com/user-attachments/assets/71e56fe9-589a-4fd5-b55e-d4f4d0073671)
![image](https://github.com/user-attachments/assets/ba188e09-0eee-48fe-bd4e-aa7039522a71)
![image](https://github.com/user-attachments/assets/36a0c472-f5ad-4545-92f0-00b1938c58e8)


---

**Reflections:**
Implementing the brain tumor detection project with MRI images was both challenging and rewarding. The process deepened my understanding of image processing and classification techniques in machine learning.

- **Understanding Image Classification:** This project enhanced my grasp of how image preprocessing affects model performance. The resizing and flattening of images were crucial for successful model input.
- **Model Evaluation:** Evaluating multiple models and their performance through cross-validation provided insights into the strengths and weaknesses of each algorithm. This reinforced the importance of model selection based on data characteristics and problem requirements.
- **SVM's Effectiveness:** The choice of SVM with an RBF kernel was validated through its superior cross-validation scores. It was fascinating to observe how kernel functions can significantly impact classification performance.
- **Kernel Trick Bammmmm!** : Today, I gained a deeper understanding of SVM, particularly how Kernel SVM operates. The kernel trick allows the model to transform the lower-dimensional feature space to a higher-dimensional space without explicitly transforming the data. Instead, it computes the dot products in the original feature space, effectively mimicking this transformation. This approach enhances efficiency in both time and space, making it a powerful technique for handling complex datasets, literally this thing is a double, triple, 100000x BAMMM!
- **Model Limitations:** While traditional models like SVM were effective in this project, I acknowledge that they may not always be the best choice compared to deep learning models like Convolutional Neural Networks (CNNs). However, for learning purposes, I opted to implement the models I have learned thus far, comparing their performance and drawing insights through visualization.
- **Practical Applications:** This project has practical implications in the medical field, where accurate and timely diagnosis is critical. The ability to automate brain tumor detection can assist medical professionals in making informed decisions.
- **Further Improvements:** I recognize that there are opportunities for improvement, such as augmenting the dataset with more images or applying advanced techniques like deep learning for potentially better performance, but currently my learning is not in that level/scope.
