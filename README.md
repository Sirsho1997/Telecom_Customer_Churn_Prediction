# Telecom Customer Churn Prediction

### The aim of this project is to apply Machine Learning to help the telecom industry to forecast customer churns to strategize marketing & business plans to retain customers.


**CHECK THIS -->** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]https://colab.research.google.com/github/Sirsho1997/Telecom_Customer_Churn_Prediction/blob/main/Telecom_Customer_Churn_Prediction.ipynb)





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

The analysis shows that there are several columns with unique values. The columns with only one unique values does not influence the prediction. For data analysis, these columns are not required and need to be dropped.



Contributor - 
- [Sirshendu Ganguly](https://www.linkedin.com/in/sirshendu-ganguly/)  [![Github](https://github.com/Sirsho1997/Book-Genre-Prediction-using-Book-Summary/blob/master/data_readme/github.png)](https://github.com/Sirsho1997)

