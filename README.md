# DS4320 Project 2: Loan Defaulting for Young Adults 


### Executive Summary 
This repository contains background information, metadata, and Python code for DS4320 Project 2 about discovering the most important features that go into loan default prediction for young adults specifically with education loans. The background information provided explains loan defaulting and the effect of educational loans. The metadata provides a data dictionary, bias and uncertainty identification, and other rationales. The pipeline contains Python code that runs through the cleaning, analysis, and visualization of the solution. 

| Spec | Value | 
| :--- | :--- | 
| Name | Iliana Vasslides | 
| NetID| fbv2sc| 
| DOI | ...| 
| Press Release | [link to press release](https://github.com/ivasslides/ds4320-project2/blob/main/press-release.md) | 
| Pipeline | ... | 
| License | ... | 

## Problem Definiton 
### General Problem 
* Loan default risk

### Specific Probloem 
* Predicting loan default risk for young adults with education loans

### Motivation 
The motivation behind focusing on loan default risk for young adults because it relates to my peers and myself the most as we work our way through college and get closer to needing to pay off loans. This problem is most related to our future and I think it is important to understand some of the reasoning and other aspects of prediction for such a big topic that can impact us. Additionally, learning more about those patterns can help students and young adults make better financial decisions and be more prepared for the future.

### Rationale 
The rationale of the refinement is focused on the fact that the young adult demographic is usually considered more vulnerable in terms of financial stability because they have less work experience and credit history to stand on. Focusing on educational loans allows us to form a more specific demographic. While young adults may struggle with all types of loans, educational loans are usually the biggest and most common type of loans that are taken out by an average adult. Concentrating on this subset of the population creates a more consistent group to study, which can improve the accuracy and usefulness of the predictive model, instead of having a broader group to study. 

### Press Release 
[The Hidden Factors Behing Young Adult Loan Defaults](https://github.com/ivasslides/ds4320-project2/blob/main/press-release.md) 

## Domain Exposition 
### Terminology 
| Term | Definition | 
|:--- | :---|
| *Loan default* | occurs when a borrower fails to make a required payment on a loan | 
| *Predictive modeling* | process of using data, statistical algorithms, and machine learning techniques to predict future outcmes based on past and current information | 
| *Feature Importance* | the degree of influence each feature has on the output or prediction made by a machine learning model | 
| *Explainable models* | type of machine learning models that provide insights into how decisions are made and address the 'black box' nature | 
| *Model accuracy* | the percentage of loan outcomes that the predictive model correctly classifies as either likely to default or likely to repay | 


### Domain 
The main domain that this project lives in is loan default risk prediction. This is a branch of finance and data analytics that focuses on estimating the likelihood that a borrower will fail to repay a loan. Financial institutions use this type of prediction to reduce losses, make lending decisions, and determine which applicants may need additional review or support. Traditional approaches often rely on factors such as income, employment history, debt levels, credit score, and past payment behavior. More recently, machine learning models have become common in this field because they can identify complex patterns across many variables and often produce more accurate predictions than simpler methods. 

### Background Readings 
[link to OneDrive folder](https://myuva-my.sharepoint.com/:f:/g/personal/fbv2sc_virginia_edu/IgA038_lf5dnRJ2KY9Ied4-fAcNxiGj93p6q-9Rs3Utl10Y?e=erLge6) 

| Title | Brief Description | Link | 
| :--- | :--- | :--- | 
| *A logistic regression model for consumer default risk* | This research paper tests a logistic regression model on credit score data with the goal of evaluating the default risk of consumer loans. The paper gives a mdoel accuracy of 89.79% and states other feature findings. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQD4yVbborOwQ5kwcG8opJfOAVMVrUq9l2tMrN_TUqAkoX8?e=2mdF98)
| *Explainable prediction of loan default based on machine learning models* | This research paper examines the impact of using machine learning models to predict loan defaults. Specifically, the paper focuses on explainable models and compares their accuracies and results. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQB8BHUVQUKLQpXZcYFEI_wzAW7eY2vr1RRaJYDle4jyPXo?e=40Eq3g) |
| *Profit-oriented loan default prediction for the financial industry* | This research paper focuses on the financial institution side of predicting loan defaults. The paper proposes a new framework that improves default prediction accuracy and profitability. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQDynOtUG2nlSY8eKK2lx4hYARvvpSIbR0zItrxy97Wr_hw?e=8ff0Rm) |
| *What is a loan default* | This article explains loan defaulting as a whole, the different types of loans, the consequences of defaulting, and ways to avoid it. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQB5YtJVNK6nQL8L-Pf8sQrzAUwJGsZh1Lw1apR3CaKH2DQ?e=P42Fsh) |
| *Why student loan borrowers are falling behind* | This article dives into the current situation of student loans and the potential negative future that is has. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/fbv2sc_virginia_edu/IQCb7Jyfm8RBTbEyhJV0BazmATqPpQTC5o5asfsE6nzRc9w?e=i839c0) |


## Data Creation 
### Raw Data Acquistion 
To start my project, I found a loan approval prediction dataset on Kaggle which included many of the features I wanted to analyze, such as age, borrower income, employment length of the borrower, loan grade, and others. Having clear columns/features is important for my feature importance analysis of the predictive models. Also, I wanted to ensure that my dataset would have enough features to create an accurate predictve model as well. The final part of my data creation was cleaning the data. I renamed some of the columns to make them more readable, I dropped rows where the person was not between 20-34 years old (to focus on young adults), and I did one-hot encoding and mapped categorical columns to numerical values to prep the dataset for predictive modelling.

### Code Used 
| File | Brief Description| Link | 
| :--- | :--- | :--- | 
| *0-pipline.md* | Markdown file containing the analysis and visualization code | ... | 
| *1-cleaning.ipynb* | Imports dataset, cleans columns, filters by age, and removes NaNs  | ... | 
| *2-analysis.ipynb* | Performs data processing, trains Logistic Regression model to predict loan status for education loans specifically, and outputs feature importance | ... | 
| *3-visualization.ipynb* | Creates a bar chart visualizing the feature importances from the analysis | ... | 
| *press-release.ipynb* | Creates the bar chart displaying the loan status of young adults, split into multiple subsection age groups, to support the Press Release | [link](https://github.com/ivasslides/ds4320-project2/blob/main/press-release.ipynb) | 
| *quant-of-uncertainty.ipynb* | Calculates mean, standard deviation, standard error, and confidence interval for all numerical features, and results are highlighted in Metadata/Uncertainty. | [link](https://github.com/ivasslides/ds4320-project2/blob/main/quant-of-uncertainty.ipynb)

### Rationale 
The rationale for filtering the dataset to only rows where `age` is between 20 and 34 years old was to minimize any selection bias that might have occurred in the sample of the population. By training the model on only data that represents 'young adults', the model is more likely to have accurate predictions for a larger population of 'young adults'. The rationale for imputing missing values was to have a more complete dataset for the model to be trained on. Removing any rows with missing data could have shrunk the dataset too small, where the model would not have sufficient data to be trained on, and so imputing is the next best way to fill these values, without introducing too much extra bias.

### Bias Identification 
Bias could be introduced into the data in two main ways: through selection bias that misrepresents young adults, and through missing data. With selection bias, the dataset could contain more rows/observations for a different subpopulation that is not young adults and that leads some demographics to be overrepresented. Even though this might not have been done intentionally by the creators of the dataset, any previous historical selection bias or overrepresentation has the ability to create bias in the data. Missing data can lead to skewed or inaccurate results because the model is not getting all possible features or information available to make the most accurate predictions. Missing data from employment length or loan interest rate could potentially skew the results and lead to more false positives or false negatives.

### Bias Mitigation 
Selection bias that misrepresents young adults can be handled by filtering the dataset to only include rows where the person's `age` is between 20 and 34 years old. This was done in the data cleaning process to help mitigate this type of bias. Bias from missing data in certain features can be accounted for by imputation, where averages of the feature are created to fill those missing values. Of course, this creates its own sort of bias, but allows the model to train on a more complete dataset.


## Metadata 
### Soft Schema 
The implicit schema below works to group the different features in such a way that makes the schema more intuitive andn readable for any users of the dataset.

{
  "age": "integer", <br>
  "income": "number",<br>
  "emp_length": "number",

"loan": {<br>
    "loan_amnt": "number",<br>
    "loan_int_rate": "number",<br>
    "loan_grade": "string",<br>
    "loan_status": "string",<br>
    "loan_percent_income": "number"<br>
  },

  "credit": {<br>
    "cred_hist_length": "number",<br>
    "default_on_file": "boolean"<br>
  },<br>

  "home_ownership": {<br>
    "MORTGAGE": "0 or 1",<br>
    "OTHER": "0 or 1",<br>
    "OWN": "0 or 1",<br>
    "RENT": "0 or 1"<br>
  },

  "loan_intent": {<br>
    "DEBTCONSOLIDATION": "0 or 1",<br>
    "EDUCATION": "0 or 1",<br>
    "HOMEIMPROVEMENT": "0 or 1",<br>
    "MEDICAL": "0 or 1",<br>
    "PERSONAL": "0 or 1",<br>
    "VENTURE": "0 or 1"<br>
  }

}

### Data Summary 
summary of database contents 
| Collection | Description | Example | 
| :--- | :--- | :--- |
| *loan_data* | Cleaned data that was used for analysis | {"_id": "69ed2d5fe12bd39801372ba4", "age": 22, "income": 59000, ... "cred_hist_length": 3} | 
| *feature_importance* | Feature importance data that was calculated at the end of analysis, for use in the visualization | {"_id": "69ed358ba5353059331ac22b", "feature": "loan_percen_income", "importance": 1.5979091184049539} | 

### Data Dictionary  
| Name | Data type | Description | Example | 
| :--- | :--- | :--- | :--- |
| *_id* | object | Mongo's autogenerated unique id | 69ed2d5fe12bd39801372ba4 |
| *age* | integer | The age of the individual in years | 22 |
| *income* | integer | The annual income of the individual, in USD | 115000 |
|*home_ownership* | string | The type of home ownership of the individual | RENT| 
| *emp_length* | integer | The length of employment, in months | 8 |
| *loan_intent* | string |  The primary purpose of the loan | EDUCATION | 
| *loan_grade* | string | A categorical variable representing the risk grade assigned to the loan, with A meanining low risk and D meaning high risk | D |
| *loan_amnt* | integer | The total amount of the loan requested by the individual, in USD | 35000 |
| *loan_int_rate* | float | The interest rate assigned to the loan | 20.25 |
| *loan_status* | integer | A binary indicator of loan repayment status, where 1 indicates loan default and 0 indicates successful repayment | 1 |
| *loan_percent_income* | float | The ratio of the loan amount to the individual's income | 0.53 |
| *default_on_file* | string | A categorical varaible indicating if the individual has previously defaulted | Y |
| *cred_hist_length* | integer | The length of the individual's credit history, in years | 3 |

### Quantification of Uncertainty 
To get quantification of uncertainty for numerical features, I calculated some basic metrics for each numerical feature. The metrics I included are mean, standard deviation, standard error, and the lower and upper bounds of a 95% confidence interval. The purpose of these metrics is to identify how spread out the data is and examine the distribution of data as well. Below is the table of metrics for each feature that was generated in the code file that is linked [here](https://github.com/ivasslides/ds4320-project2/blob/main/quant-of-uncertainty.ipynb).
| Feature | Mean | Std_dev | Standard_error| CI interval|
|:--- | :--- | :--- | :--- | :--- |
| age	| 25.89 |	3.44| 0.02 |25.85- 25.93|
| income	|64149.33	|41883.03	| 248.12| 	63663.02- 64635.63|
| emp_length|4.61|3.76|	0.02|4.57-4.65|
|loan_grade|	2.21|	1.16|	0.01|	2.20-2.22|
|loan_amnt|	9536.37	|6271.80	|37.15|	9463.55-9609.19|
|loan_int_rate	|10.99|	3.08|	0.02|	10.96-11.03|
|loan_status	|0.22|	0.41|	0.00|	0.21-0.22|
|loan_percent_income|0.17|0.11|0.00|0.17-0.17|
|default_on_file|0.18|	0.38|	0.00	|0.17-0.18|
|cred_hist_length	|4.68|	2.50|	0.01	|4.65-4.71|
