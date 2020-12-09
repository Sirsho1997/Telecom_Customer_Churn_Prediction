# Telecom Customer Churn Prediction

## The aim of the project is to help the Telecom industry to forecast customer churns in order to strategize marketing and business plans to retain customers.

   IPython Notebook has been used for this project.
   
   
   - Not uploading the data set for privacy. 
   
   - The goal of this repository is to provide an example of a analysis for those interested in getting into the field of Data Science.
   
   - Required Libraries
       - [Pandas](https://pandas.pydata.org/)
       
       - [NumPy](https://numpy.org/")

       - [Scikit-Learn](https://scikit-learn.org/stable/)

       - [Matplotlib](https://matplotlib.org/)

       - [Seaborn](https://seaborn.pydata.org/)

       


### For viewing the whole code - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/Telecom_Customer_Churn_Prediction.ipynb)


 
 #### Data Analysis
 
- Let us first have a look at information associated with the data set.


<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/telecomoverview.png" />

- Next, let us count the the number of unique values for each of the columns.


<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/unique.png" />


#### Remove the columns with unique values

- The above analysis depicts that there are several columns with unique values. 
The columns with unique values does not influence the task of prediction. Thus these columns are dropped.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/removeunique.png"  />

- Plotting the unique values of PROD_OFR_KEY

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/PROD_OFR_KEY.png" />

- Applying binning on PROD_OFR_KEY 

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/binning.png" />

- Since '0' is just 0.58% , so it would be wise to discard the rows containing those values.


- Plotting the unique values of PROD_LN_CD

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/PROD_LN_CD.png" />

- As the above analysis indicates that the column PROD_LN_CD contains unique value, thus dropping it.

- Mobile number is of data type object and value is different for each row. Mobile number column is not useful for training the model.Thus extracting the mobile number for later use.


#### Missing data imputation

- Let us first have a visual of the heatmap of the data set to depict a pictorial representation of the missing values.


<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/heatmap.png" width="50%" height="60%" />

- Counting the number of missing values 

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/missingvalues.png" />

- If we look at the rows where TOT_DAYS_ACTVTY and TOT_DAYS_OUTGOING_ACTVTY is NaN, then we find that all the other column values are same, thus discarding those rows.

- Plotting the distplot for DAYS_BFR_FIRST_RCHRG.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/distplot.png" width="50%" height="60%" />

- Performs Random Imputation followed by Regression Imputation to impute the missing values for DAYS_BFR_FIRST_RCHRG.
  

#### Checking for Correlation

- Let us look at at the scatter plot between TOT_CALL_CNT_LAST_MO and TOT_CALL_CNT_LAST_3MO.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/scatterTOT_CALL.png" width="50%" height="60%" />

The above analysis indicates that there is a strong positive correlation between the TOT_CALL_CNT_LAST_MO and TOT_CALL_CNT_LAST_3MO. Last 3 months data is linear aggregation of last one month and the data is contained in it. In view of this we can drop the column "Total number of calls made in last month. i.e Column titled'TOT_CALL_CNT_LAST_MO'.

- Now let us look at at the scatter plot between TOT_TALK_DRTN_LAST_MO and TOT_TALK_DRTN_LAST_3MO.

<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/scatterTOT_TALK_DRTN_LAST_3MO.png" width="50%" height="60%" />

The above analysis indicates that there is a strong positive correlation between the TOT_TALK_DRTN_LAST_MO and TOT_TALK_DRTN_LAST_3MO. Last 3 months data is linear aggregation of last one month and the data is contained in it. In view of this we can drop the column titled'TOT_TALK_DRTN_LAST_MO'.

- Following in this pattern , we remove all correlation among data points.

#### Checking for Skewness in the Data Set

- The columns which are left skewed data are transformed using square() function.
- The columns which are right skewed are transformed using sqrt() function.
- Performing Normalization.


#### Fitting Model

##### Logistic Regression
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/LR.png"  />

##### KNN 
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/KNN.png"  />

##### Decision Tree
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/DT.png"  />

##### Random Forest
<img src="https://github.com/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/image/RF.png"  />


Contributor - 
- [Sirshendu Ganguly](https://www.linkedin.com/in/sirshendu-ganguly/)  [![Github](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/github.png)](https://github.com/Sirsho1997)

