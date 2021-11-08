---
layout: post
title:      "Let’s Reduce Customer Churn!"
date:       2019-03-04 20:29:25 -0500
permalink:  why_did_you_decide_to_learn_data_science
---

#### SyriaTel Churning Analysis 

Natalia Edelson

![](https://i.imgur.com/aJVNKcO.png/)


Telecom companies have been facing the increasing  challenge of customer’s attrition. Telecom companies are focused on predicting customer churn in order to avoid a major fall in their revenue. It’s often the case that onboarding a new customer is more costly than retaining an existing client. For the purpose of this case study, we will gather data of the telecom company SyriaTel and our analysis will be centered around the possible ways it can reduce its churn.    


SyriaTel had seen 16% of customers leave their business.  We built an analysis using Python with Scikit-Learn to find the important factors contributing to customer churn the most. We built a predicting model to allow us to obtain the insight of the features that should be closely monitored in order to reduce customer churn in SyrianTel.



We consider the telecom challenge as a classification problem in which we predict whether a customer will churn (1) or not (0). We will use machine learning methods to build out models to find the features of importance.

We obtained the data from Kaggle and follow the below steps:

1.	Preform a data cleaning.
2.	Explore the data – we look for trends using statistical methods. 
3.	Build classified models – Logistic Regression, k-nearest neighbors (k-NN), DecisionTree and XGBoost. We will tune the models as well aiming to get optimal results. 
4.	Examine the features of impotence to interpret results and put them to use.  



### Cleaning the Data
```
#Check for null values
null_counts = Customer_Churn.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))

Number of null values in each column:
state                     0
account_length            0
area_code                 0
phone_number              0
international_plan        0
voice_mail_plan           0
number_vmail_messages     0
total_day_minutes         0
total_day_calls           0
total_day_charge          0
total_eve_minutes         0
total_eve_calls           0
total_eve_charge          0
total_night_minutes       0
total_night_calls         0
total_night_charge        0
total_intl_minutes        0
total_intl_calls          0
total_intl_charge         0
customer_service_calls    0
churn                     0
dtype: int64
In [6]:



#Check for duplicates

Customer_Churn.duplicated().sum()
Out[]:0
 

#Explore the dataset's stats and check for outliers 
display(Customer_Churn.describe())
```





![](https://imgur.com/eYPcNof) 


We compare the mean versus the max and min values in order to sort outliers or potential mistakes. There are no outliers in this data. Overall, the data is clean. It doesn’t have missing values nor unnecessary fillers. (In the coding on Github one can find more details on the unhelpful variables we chose to omit).




### Exploring the Data


In a classified problem it is important to check whether the data is imbalanced. When we start building our model it will be key to take this into account when evaluating the results. 


![](https://imgur.com/eYPcNof)

We can see the data is not balanced as 85% of people are not churning.

Below we check the correlation between variables, and we will examine the ones that show a strong correlation. 

```
sns.set(style="white")

corr = Customer_Churn.corr().round(2)

mask = np.triu(np.ones_like(corr, dtype=bool))
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(14, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title(' Costumer Churn Variables - Correlation Matric Hat Map ')
```



![](https://imgur.com/UR8FNBW)


We can clearly see the higher number of Customer Service calls will likely lead to a customer leaving. Particularly after three calls, we saw an increase in churning. 




In the total day charge, we can see that customers are much more likely to churn right after the $38 day charge. Thus, this is an area of concern for SiryaTel.

We saw a similar pattern in the evening charge but with a more concentrated dollar amount. Roughly speaking, customers who were charged for the evening calls are much more likely to churn. 


