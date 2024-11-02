___
## Day 31
### Topic: Implementation of GaussianNB, MultinomialNB, BernoulliNB
*Date: November 02, 2024*

### Detailed Notes 📝

Today, I practiced implementing different types of Naive Bayes classifiers using `scikit-learn` in Python. Each Naive Bayes variant works best with specific types of data, so I experimented with different datasets to understand how each classifier functions in its optimal scenario.

### 1. Gaussian Naive Bayes (Continuous Data)

Gaussian Naive Bayes is ideal for continuous data because it assumes a normal distribution for each feature. I used the **Iris** dataset, which contains continuous features like sepal and petal measurements.


![image](https://github.com/user-attachments/assets/22069e8b-25f6-44e0-836a-2d06c0f695a4)


**Note**: The continuous nature of the Iris dataset made Gaussian Naive Bayes a good fit, as it assumes the data follows a Gaussian distribution.

---

### 2. Multinomial Naive Bayes (Discrete Data)

Next, I explored the Multinomial Naive Bayes classifier, which is best for discrete data, such as word counts in text classification. For this, I used the **20 Newsgroups** dataset, where features represent word counts.


![image](https://github.com/user-attachments/assets/a120fe9a-99bb-482b-b089-a7bf0cea3385)


**Note**: Multinomial Naive Bayes works well with this data because it uses word counts to estimate probabilities, which makes it ideal for text data where we’re looking at the frequency of words.

---

### 3. Bernoulli Naive Bayes (Binary Data)

For the final experiment, I used Bernoulli Naive Bayes. This classifier is designed for binary features, so I transformed the word counts from the 20 Newsgroups dataset features into binary.


![image](https://github.com/user-attachments/assets/92ca87d4-63fb-4bcd-a3b8-d8fdf5443f3e)


**Note**: Since each feature is now binary, Bernoulli Naive Bayes is an excellent choice. It calculates probabilities based on whether each feature is present (1) or absent (0), making it ideal for binary data.

---

### My Final Thoughts

1. **Gaussian Naive Bayes** - I used this for continuous data in the Iris dataset, as it assumes features follow a Gaussian distribution.
2. **Multinomial Naive Bayes** - This was ideal for the 20 Newsgroups dataset with discrete word counts, as it calculates probabilities based on frequencies.
3. **Bernoulli Naive Bayes** - Worked best for binary data (binary word presence in 20 Newsgroups), where features are either 0 or 1.

#### 1. **Bernoulli Naive Bayes**
   - **Use Case**: When data consists of **binary features** (each feature is either 1 or 0).
   - **Example**: Text classification tasks where each word is either **present (1)** or **absent (0)** in a document.
   - **Best Example**: Spam detection. Here, each word in an email can be treated as a feature that either appears (1) or does not appear (0). Bernoulli Naive Bayes can model the likelihood of the email being spam based on the presence or absence of certain keywords.

---

#### 2. **Multinomial Naive Bayes**
   - **Use Case**: When data consists of **discrete data** i.e. **count data** or **frequency of events**.
   - **Example**: Text classification tasks where each feature represents **the frequency or count of each word** in a document.
   - **Best Example**: Document classification (e.g., classifying articles by topic). Here, words can appear multiple times, and the model leverages this frequency information to determine the topic based on words that appear often within that category (like “rocket” or “space” for space-related topics).

---

#### 3. **Gaussian Naive Bayes**
   - **Use Case**: When features are **continuous** and **normally distributed** (follows a Gaussian or bell-shaped curve).
   - **Example**: Classification tasks involving **numeric features** like height, weight, or age.
   - **Best Example**: Iris flower classification. The Gaussian Naive Bayes model can be used to classify types of iris flowers based on the continuous features (like petal length, petal width, etc.) which are often normally distributed across species.

---

### Summary Table:

| Classifier          | Best for Data Type     | Real-World Example             |
|---------------------|------------------------|---------------------------------|
| **Bernoulli NB**    | Binary features        | Spam detection                 |
| **Multinomial NB**  | Counts or frequencies (Discrete Data)  | Document classification        |
| **Gaussian NB**     | Continuous values      | Iris flower classification      |

Each of these Naive Bayes classifiers assumes feature independence but handles different types of data, making it versatile for many types of classification tasks.

These experiments showed me how choosing the right Naive Bayes classifier depends on the feature types in the dataset, and how each classifier leverages different data characteristics for effective classification.
