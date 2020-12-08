# Telecom Customer Churn Prediction

### The aim of this project is to apply Machine Learning to help the telecom industry to forecast customer churns to strategize marketing & business plans to retain customers.


**CHECK THIS -->** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/Telecom_Customer_Churn_Prediction.ipynb)





- The dataset that have been used here is 




 ### Rough insight of the work
 
 ##### Data Analysis
 
- Let us first have a look at data set.

```python
#Data Overview
print ("Rows     : " ,telcom.shape[0])
print ("Columns  : " ,telcom.shape[1])
print ("\nFeatures : \n" ,telcom.columns.tolist())
```

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/telecomoverview.png" width="50%" height="60%" />

- Next let us look at the number of unique values.

```python
#Finding the number of unique values for each feature
telcom.nunique()
```

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/unique.png" width="50%" height="60%" />


##### Remove the columns with unique values

- The analysis shows that there are several columns with unique values. The columns with only one unique values does not influence the prediction. For data analysis, these columns are not required and need to be dropped.


```python
#Iterate through the columns and form a list of columns where mean = max = min and std = 0
cols_with_zero_values = []
count=0

for cols in telcom.columns:
  if telcom[cols].min() == telcom[cols].max() and telcom[cols].mean() == telcom[cols].max() and telcom[cols].std() == 0:
    cols_with_zero_values.append(cols)
    count=count+1

  print("The number of columns with min = mean = max ",count)
  telcom = telcom.drop(cols_with_zero_values, axis=1)
```
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/removeunique.png" width="50%" height="60%" />

- Plotting the unique values of PROD_OFR_KEY
```python
plt.figure(figsize=(16,4))
ax = sns.countplot(x=telcom['PROD_OFR_KEY'], data=telcom)
```
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/PROD_OFR_KEY.png" width="50%" height="60%" />

- Applying binning
```python
# Applying binning

def transform_PROD_OFR_KEY(PROD_OFR_KEY):
    x = [20150818,20140810]
    if PROD_OFR_KEY not in x:
        return 0
    else:
        return PROD_OFR_KEY
    
telcom['PROD_OFR_KEY'] = telcom['PROD_OFR_KEY'].apply(transform_PROD_OFR_KEY)
telcom['PROD_OFR_KEY'].value_counts()
```
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/binning.png" width="50%" height="60%" />

- Since '0' is just 0.58% , so it would be wise to discard the rows containing those values.

```python
telcom = telcom[telcom['PROD_OFR_KEY']!=0]
```

- Plotting the unique values of PROD_LN_CD
```python
plt.figure(figsize=(16,4))
ax = sns.countplot(x=telcom['PROD_LN_CD'], data=telcom)
```
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/PROD_LN_CD.png" width="50%" height="60%" />

- As the values of feature 'PROD_LN_CD' contains only one unique value so removing it.

```python
telcom = telcom.drop(['PROD_LN_CD'], axis=1)
```
- Mobile number is of data type object and value is different for each row. Mobile number column is not useful for training the model. But later mobile number is needed for predicting the churn with probability.Thus extracting the mobile number for later use.


##### Missing data imputation

- Let us first have a visual of the heatmap of the data set.

- Counting the number of missing values 
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/heatmap.png" width="50%" height="60%" />

```python
#Printing the count of missing values from each of the columns in the DataFrame

for col in telcom.columns:
    if telcom[col].isna().sum() > 0:
        print(col,'=>',telcom[col].isna().sum())
```

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/missingvalues.png" width="50%" height="60%" />

- If we look at the rows where TOT_DAYS_ACTVTY and TOT_DAYS_OUTGOING_ACTVTY is NaN, then we find that all the other column values are same, thus discarding those rows.

- Plotting the distplot for DAYS_BFR_FIRST_RCHRG.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/distplot.png" width="50%" height="60%" />

- Performs Random Inputation followed by Regression Imputation to impute the missing values for DAYS_BFR_FIRST_RCHRG.

##### Checking for Correlation

- Let us look at at the scatter plot between TOT_CALL_CNT_LAST_MO and TOT_CALL_CNT_LAST_3MO.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/scatterTOT_CALL.png" width="50%" height="60%" />

The above analysis indicates that there is a strong positive correlation between the TOT_CALL_CNT_LAST_MO and TOT_CALL_CNT_LAST_3MO. Last 3 months data is linear aggregation of last one month and the data is contained in it. In view of this we can drop the column "Total number of calls made in last month. i.e Column titled'TOT_CALL_CNT_LAST_MO'

- Now let us look at at the scatter plot between TOT_TALK_DRTN_LAST_MO and TOT_TALK_DRTN_LAST_3MO.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/images/scatterTOT_TALK_DRTN_LAST_MO.png" width="50%" height="60%" />

The above analysis indicates that there is a strong positive correlation between the TOT_TALK_DRTN_LAST_MO and TOT_TALK_DRTN_LAST_3MO. Last 3 months data is linear aggregation of last one month and the data is contained in it. In view of this we can drop the column "Total number of calls made in last month. i.e Column titled'TOT_TALK_DRTN_LAST_MO'

- Following in this pattern , we remove all correlation among data points.









Contributor - 
- [Sirshendu Ganguly](https://www.linkedin.com/in/sirshendu-ganguly/)  [![Github](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/github.png)](https://github.com/Sirsho1997)

