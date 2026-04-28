# Loan Default Prediction for Young Adults with Education Loans 

*Starting the pipeline from the analysis section, since the data that is stored in MongoDB is already cleaned and ready for processing.* 

## Analysis 

```python 
import pandas as pd 
from pymongo import MongoClient 
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging 

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='analysis.log'
)
logger = logging.getLogger(__name__)
``` 

### Pull data from Mongo DB 
Before starting analysis, we need to pull the records from MongoDB using the database user and password stored in the environment variables. 

```python
# retrieve credentials from env 
load_dotenv()
DB_USER = os.getenv('DB_USER')
PASSWORD = os.getenv('PASSWORD')

# make connection 
client = MongoClient(f'mongodb+srv://{DB_USER}:{PASSWORD}@ds4320-project2.lq6desq.mongodb.net/?appName=ds4320-project2')
db = client['ds4320_project2']
collection = db['loan_data']

data = list(collection.find()) 
df = pd.DataFrame(data)

# drop MongoDB’s automatic _id field
df = df.drop(columns=["_id"])
df
logging.info('Data loaded into pandas df successfully')
```

Also before starting the analysis, some of the variable need to be one-hot encoded. This is because they are categorical variables, but a Logistic Regression model can't process those as well as numeric values. `home_ownership` and `loan_intent` will be one-hot encoded, in addition to `loan_grade` being mapped to numerical values that correspond with the letter value. `default_on_file` needs to be numerically mapped as well, because right now its values are 'Y' and 'N'.

```python
# one hot encode person_home_ownership and drop original column 
df2 = df.copy() 
try: 
    df2 = pd.get_dummies(df2, columns=['home_ownership', 'loan_intent'], dtype=int)

    # map loan grade to numerical values 
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df2['loan_grade'] = df2['loan_grade'].map(grade_map)

    # map previous default to 1 for yes and 0 for no
    df2['default_on_file'] = df2['default_on_file'].map({'Y': 1, 'N': 0})
    
    logger.info('Data one-hot encoded and mapped successfully.')
    print(df2)
except Exception as e:
    logger.error(f'Error: {e}')
```

### Model training 
To keep the focus on educational loans as discussed previously, a subset of the dataframe is created to only contain observations where the loan intent is educational. Additionally, we are dropping the other loan intent columns because they just contain 0s.

```python 
# creating subset df for education loan intent
df_ed = df2[df2['loan_intent_EDUCATION'] == 1].copy().reset_index(drop=True) 
df_ed = df_ed.drop(columns=['loan_intent_EDUCATION', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'])
df_ed
```
To start the model training and analysis, a train-test split is done to increase model accuracy and generalization. Scaling of the X values are also done to ensure that every feature is scaled the same, so the model can evaluate them as accurately as possible.

```python 
# create train test split 
X = df_ed.drop(columns=['loan_status'])
y = df_ed['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print shapes to verify 
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}') 
```

```python 
# scaling the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# making predictions 
y_pred = model.predict(X_test_scaled)
logger.info('Logistic regression model trained and predictions made.')
```

To evaluate the model's performance, four metrics are calculated. Accuracy, precision, recall, and f1 score are common metrics to evaluate a Logistic regression model. Furthermore, we will look at the feature importance from the model by using the coefficients. This shows uthe impact that each feature had on the model's decisions.

```python 
# calculating error metrics 
accuracy = round(accuracy_score(y_test, y_pred),4) 
precision = round(precision_score(y_test, y_pred),4)
recall = round(recall_score(y_test, y_pred),4)
f1 = round(f1_score(y_test, y_pred),4)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
```

```python 
# create feature importance from model 
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.coef_[0]
}).sort_values(by='importance', ascending=False)

print(feature_importance)
```

### Save to Mongo 
Here we are saving the feature importance dataframe to MongoDB so that it can be queried later on. 

```python
# export data to json file 
feature_importance.to_json('feature_importance.json', orient='records', lines=True)
logger.info('Feature importance exported to JSON successfully.')
```

Mongosh code - run in terminal 

    
    use ds4320_project2
    
    const fi = require("./feature_importance.json")
    
    db.feature_importance.insertMany(fi)


```python
logger.info('Feature importance saved to MongoDB successfully.')
```

## Visualization 
```python
import matplotlib.pyplot as plt 

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='visualization.log'
)
logger = logging.getLogger(__name__)
```

### Pull data from MongoDB 
```python 
# retrieve credentials from env 
load_dotenv()
DB_USER = os.getenv('DB_USER')
PASSWORD = os.getenv('PASSWORD')

# make connection 
client = MongoClient(f'mongodb+srv://{DB_USER}:{PASSWORD}@ds4320-project2.lq6desq.mongodb.net/?appName=ds4320-project2')
db = client['ds4320_project2']
collection = db['feature_importance']
logging.info('MongoDB connected!')

# pulling feature importance data 
fi = list(collection.find()) 
fi = pd.DataFrame(fi)

# drop MongoDB’s automatic _id field
fi = fi.drop(columns=["_id"])

logging.info('Data loaded into pandas df successfully')
fi
```

### Create graph 
For this visualization, a bar chart was the best option because we want to compare the values, aka the heights, of each bar to that of the other bars. Additionally, being able to compare negative values to the positive values is important to determine if those features had a positive or negative impact on the prediction decision.

```python 
# making bar chart of feature importance
fig, ax = plt.subplots(figsize=(14,7))
ax.bar(fi['feature'], fi['importance'])

# create more readable feature names
ax.set_xticks(ticks=range(len(fi['feature'])), labels=['Loan\nPercent\nIncome', 'Loan\nGrade', 'Home\nOwnership\nRent', 'Age', 'Income', 'Loan\nInterest\nRate', 
                    'Home\nOwnership\nOther', 'Employment\nLenght', 'Default\non\nFile', 'Credit\nHistory\nLenght', 'Home\nOwnership\nMortgage', 
                    'Home\nOwnership\nOwn', 'Loan\nAmount'])

ax.axhline(0, color='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title("Feature Importance for Predicting Loan Default of Educational Loans of Young Adults", fontsize=18)
ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Importance", fontsize=14)
plt.text(6, -1.7, """This bar chart is a visualization of the feature importance values obtained from a logistic regression model trained to predict loan default 
        for educational loans taken out by young adults, ages 20-34. The features are ranked based on their impact in the model's prediction of  
        whether the individual will default on the loan or not. Positive importance values indicate that the feature increases the likelihood   
        of defaulting, while negative values indicate that the feature decreases the likelihood of defaulting.""", ha='center')

# save as png 
plt.savefig('visualization.png', bbox_inches='tight')
plt.show()
logging.info('Feature importance bar chart created and saved')
``` 