___
## Day 10
### Topic: Multiple Linear Regression in Python
*Date: October 12, 2024*

**Today's Learning Objectives Completed ✅**
- Mastered Multiple Linear Regression in Python with sklearn
- Trained a model to predict startup profits based on multiple features
- Visualized model performance using KDE plots
- Integrated the trained model into a Next.js frontend with a Flask backend into a [VC Profit Predictor](https://vc-profit-predictor.vercel.app/) web application

### Detailed Notes 📝

**Multiple Linear Regression Implementation**

I implemented a multiple linear regression model to predict startup profits based on various features:
- R&D Spend
- Administration Spend
- Marketing Spend
- State (categorical feature)

Key steps in the implementation:

a) Data Preprocessing:
```python
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

b) Handling Missing Values:
```python
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])
```

c) Encoding Categorical Data:
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

d) Train-Test Split:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

e) Model Training:
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

**Model Visualization and Evaluation**

I used KDE plots to visualize the model's performance on unseen data:

```python
sns.set_theme(style="darkgrid")
sns.kdeplot(y_test, color="red", label="Actual Values")
sns.kdeplot(y_hat, color="blue", label="Fitted Values")
plt.title("Actual v/s Fitted Values (Test Set)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/16bc858c-6110-4472-972e-45df936c24a0)

![image](https://github.com/user-attachments/assets/477ad37a-6504-40dd-90cf-ce666ac05d6e)


The KDE plot shows a close alignment between actual and predicted values, indicating good model performance.

**VC Profit Predictor Web Application**

I integrated the trained model into a web application using Next.js for the frontend and Flask for the backend.

**Key features of the application:**
- Input fields for R&D Spend, Administration Spend, Marketing Spend, and State
- Prediction of potential profit based on input values
- Sample data buttons for quick testing
- Clear explanation of the application's purpose and usage

**Screenshots of the application:**

*The main interface of the VC Profit Predictor, showing input fields and prediction result*

![image](https://github.com/user-attachments/assets/e9a17e7c-6568-4091-8aa3-73806e62d940)

*Sample data feature for quick testing of different scenarios*

![image](https://github.com/user-attachments/assets/ec34b64c-967f-4923-93f2-1784741ae975)

**Key Insights 💡**
- Multiple linear regression allows us to consider various factors affecting startup profitability
- The model shows good performance on unseen data, as visualized by the KDE plot
- Integrating ML models into web applications provides an accessible interface for non-technical users
- This tool can help VCs make data-driven investment decisions by analyzing spending patterns and regional variations
- This project demonstrates the practical application of machine learning in a business context, showcasing how data science can inform investment strategies in the venture capital world.
