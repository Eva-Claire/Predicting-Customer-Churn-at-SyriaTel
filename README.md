# SYRIATEL CUSTOMER CHURN PREDICTION: LEVERAGING DATA TO IMPROVE CUSTOMER RETENTION

![logo.png](attachment:logo.png)

#### Author: Evaclaire Munyika Wamitu  
#### Student pace: Remote  
#### Technical mentors: Asha Deen / Lucille Kaleha  


Welcome to the "SYRIATEL CUSTOMER CHURN PREDICTION" project! Dive into our in-depth analysis of customer churn for Syriatel, a telecom giant looking to uncover the crucial factors driving subscriber turnover. Our mission? To equip Syriatel and telecom businesses worldwide with actionable insights and retention strategies, ensuring every customer feels valued and satisfied. Join us as we decode churn and pave the way for long-lasting customer relationships.

# OVERVIEW

SyriaTel faces a critical challenge: identifying customers likely to cancel services, a common issue in the telecom industry affecting revenue. Leveraging advanced predictive analytics and machine learning, We aim to build a robust churn prediction model. This will empower SyriaTel to proactively engage at-risk customers, implement retention strategies and mitigate financial losses. The project's main stakeholder is SyriaTel, company executives, marketing teams and customer service reps keen on reducing churn and maximizing customer value. By uncovering churn factors and identifying at-risk customers, the model enables data-driven decisions, optimizing retention efforts for sustained business growth.


# OBJECTIVE

The main objective of this project is to develop a binary classification model that can accurately predict whether a customer will stop doing business with SyriaTel in the near future. 


# METHODOLOGY

1. **Data Collection and Preprocessing** : This involves gathering the relevant customer data that will aid us in uncovering factors influencing churn.
2. **Exploratory Data Analysis (EDA)** : Conducting EDA to gain insights, identify patterns and visualize relationships between features and customer churn.
3. **Feature Engineering** : Selecting and engineering relevant features from the data to improve churn prediction.
4. **Model Development** : Employing machine learning algorithms like logistic regression, decision trees, random forests and K-Nearest Neighbors for binary classification.
5. **Model Evaluation and Validation** : Model evaluation metrics including accuracy score, precision, recall, F1-score and area under the ROC curve (AUC-ROC). Models will be validated using cross-validation.
6. **Interpretation and Deployment** : Interpreting the trained model and deploying it into SyriaTel's production environment
7. **Monitoring and Evaluation** : Retraining/updating of the model to ensure ongoing effectiveness.


# DATA UNDERSTANDING

### Data Collection
The dataset used in this project is called Churn in Telecom's from Kaggle (https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset).

The dataset consists of 3333 entries and 21 columns namely state, account length, area code, phone number, international plan, voicemail plan, number of voicemail messages, total day minutes, total day calls, total day charge, total evening minutes, total evening calls, total evening charge, total night minutes, total night calls, total night charge, total international minutes, total international calls, total international charge, customer service calls, and churn status.


# DATA PREPARATION

### Data Cleaning

- **Handle Missing Values**: Identify and decide on a strategy to handle missing values in columns.
- **Remove Duplicates**: Identify and remove duplicate rows from the dataset.
- **Address Outliers**: Detect and decide on how to handle outliers in numerical features.
- **Standardize Data Formats**: Ensure consistency in data formats across different columns.
- **Encoding Categorical Variables**: Convert categorical variables into numerical format using techniques like one-hot encoding and label encoding
- **Scaling Numerical Features**:Scale numerical features to a similar range to prevent dominance by features with larger magnitudes.
- **Feature Selection**: Evaluate the relevance of each feature to the prediction task and remove irrelevant or redundant features 
- **Handling Imbalanced Classes**: Applying techniques such as oversampling, undersampling or SMOTE for imbalanced datasets.


# EXPLORATORY DATA ANALYSIS

### Univariate data analysis 

##### Histograms of numeric features with kernel density estimation (KDE)
![png](output_34_0.png)
    
    
##### Boxplots displaying outliers

![png](output_39_0.png)
    
    
##### Countplot displaying class imbalance
  
  
![png](output_42_0.png)
    
    
##### Distribution of churn in two categorical groups


![png](output_46_0.png)
    

![png](output_46_1.png)

    
##### Comparative pairplots 

![png](output_49_0.png)
    

### Multivariate Analysis

   
![png](output_64_0.png)
    

# MODELLING


### Logistic Regression
  
![png](output_91_1.png)
     
![png](output_91_2.png)
    
After hyperparameter tuning, best parameters = {‘C’: 0.1, ‘penalty’: ‘L1’, ‘solver’: ‘liblinear’}, train accuracy of 78% and test accuracy of 78%.


### K-Nearest Neighbors


![png](output_99_1.png)
     
![png](output_99_2.png)
    

After hyperparameter tuning, best parameters = \( k = 3 \) neighbors, \( p = 1 \) (indicating Manhattan distance)


### Decision Trees Classifier


![png](output_106_1.png)
    
![png](output_106_2.png)   

After hyperparameter tuning, train accuracy went down from 89% to 93% and test accuracy from 91% to 82%. 

![png](output_108_0.png)
    

### Random Forest Classifier

![png](output_111_1.png)
    
  
![png](output_111_2.png)
    
Training accuracy of 100% and a test accuracy of 93%.


# MODEL EVALUATION

![png](output_117_0.png)
    

![png](output_121_0.png)
    

# CONCLUSIONS

- **Call Usage Features**: Customers' calling behavior, including duration and frequency, emerges as strong predictors of churn, indicated by high importance scores.
  
- **Customer Tenure and Service Characteristics**: Metrics like 'account_length' and service usage patterns play a significant role in predicting churn, reflecting the impact of customer tenure and service usage.

- **Geographical Location**: While 'area_code' suggests potential influence on churn, state-level features exhibit lower importance scores, indicating lesser predictive power.

- **Feature Selection and Model Optimization**: Considering the model's ability to handle numerous features, optimizing performance and interpretability may benefit from feature selection or dimensionality reduction techniques.


# RECOMMENDATIONS

- **Primary Predictive Model: Random Forest Classifier**: Leveraging the Random Forest Classifier offers exceptional performance with impressive metrics like ROC curve, accuracy, F1-score, recall and precision, making it ideal for accurately classifying potential churners and loyal subscribers.

- **Monitoring Call Usage Patterns**: SyriaTel should closely track call usage behaviors and implement targeted retention strategies for customers showing high call usage or significant changes in their calling behavior.

- **Improving Customer Satisfaction and Loyalty Programs**: Enhancing customer satisfaction and offering attractive loyalty programs can mitigate churn risk, especially among long-term customers.

- **Regional Analysis and Tailored Strategies**: Analyzing regional churn differences enables SyriaTel to customize marketing and customer service strategies, catering to region-specific needs and preferences.

- **Feature Prioritization and Model Improvement**: While state-level features remain relevant, prioritizing critical features like call usage and customer tenure enhances churn prediction accuracy. Techniques like recursive feature elimination and exploring other machine learning algorithms can further optimize model performance.
