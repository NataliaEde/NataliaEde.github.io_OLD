---
layout: post
title:      "Let’s Reduce Customer Churn!"
date:       2019-03-04 20:29:25 -0500
permalink:  why_did_you_decide_to_learn_data_science
---

#### SyriaTel Churning Analysis 

Natalia Edelson

![](https://i.imgur.com/saMLNzB.jpg)


Telecom companies have been facing the increasing  challenge of customer’s attrition. Telecom companies are focused on predicting customer churn in order to avoid a major fall in their revenue. It’s often the case that onboarding a new customer is more costly than retaining an existing client. For the purpose of this case study, we will gather data of the telecom company SyriaTel and our analysis will be centered around the possible ways it can reduce its churn.    


SyriaTel has seen 14.5% of customers leave their business.  We built an analysis using Python with Scikit-Learn to find the important factors contributing to customer churn the most. We built a predicting model to allow us to obtain the insight of the features that should be closely monitored in order to reduce customer churn in SyrianTel.

![](https://i.imgur.com/IFLs63A.png?2)

We consider the telecom challenge as a classification problem in which we predict whether a customer will churn (1) or not (0). We will use machine learning methods to build out models to find the features of importance.

We obtained the data from Kaggle and follow the below steps:

1.	Preform data cleaning.
2.	Explore the data – we look for trends using statistical methods. 
3.	Build classified models – Logistic Regression, k-nearest neighbors (k-NN), DecisionTree and XGBoost
4.	We will tune the models as well aiming to get optimal results. 
5.	Examine the features of impotence to interpret results and put them to use.  



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



![](https://i.imgur.com/2AbuItr.png)

We can clearly see the higher number of Customer Service calls will likely lead to a customer leaving. Particularly after three calls, we saw an increase in churning. 



![](https://i.imgur.com/yVjPQTr.png)


In the total day charge, we can see that customers are much more likely to churn right after the $38 day charge. Thus, this is an area of concern for SiryaTel.


![](https://i.imgur.com/7ru6JnW.png)


We saw a similar pattern in the evening charge but with a more concentrated dollar amount. Roughly speaking, customers who were charged above $18 for the evening calls are much more likely to churn. 


![](https://i.imgur.com/MGA6Vtd.png)




**Building Classified Models**

Our supervised learning task is a classification problem and therefore we will be labeling the data and then scaling it. We create ‘x’ and ‘y’ by selecting 'churn' from the dataset and then we create an 80/20 split on the dataset for training/test. We use random_state=10 to achieve reproducible results. 
We scale the data using the Standard Scaler method and standardize the data by making the mean of the distribution zero and the majority of the data will be between -1 and 1.
  
We will be using the following supervised learning algorithm.  While building our models we will be utilizing GridsearchCV, which is an exhaustive search technique to help us find optimal combination of hypermeters. 

We train the models below and compare their performance. 

Logistic Regression
k-nearest neighbors (k-NN), 
DecisionTree 
XGBoost.


We then measure our performance by looking into various scores. We will investigate the following: 

```
Train_Accuracy = model.score(X_train_scaled,y_train)
Test_Accuracy = model.score(X_test_scaled,y_test)
Precision = precision_score(y_test,y_preds)
Recall = recall_score(y_test,y_preds)
f1_Score = f1_score(y_test,y_preds)

np.mean(cross_val_score(model, X_scaled, y, scoring="recall", cv = 5)))


```

We also utilize the cross-validation score – which uses five different validation sets to average out and provide us with a more accurate measurement of performance. Specifically, we will be focusing on the recall score because we do not want to miss a false negative. If a customer left and we missed that data, it could be very costly for SyriaTel. We have a lesser need for precision because in the worst-case scenario we could offer a customer whom we mistakenly thought had left an incentive to stay with the company.

We compare separately how well the two different classes (churn or no churn) were predicted by using classification reporting. 

We can’t rely on accuracy because it gives us deceiving results as our data is imbalanced. 

![](https://i.imgur.com/orIr6nN.png)

The highest recall results were in XG Boost. 

![](https://i.imgur.com/4CrUtjg.png)

We investigate which features have the most impact on the accuracy of our trained model XG Boost. 

![](https://i.imgur.com/VsSZ8D9.png)


**Features of Importance**

**Voicemail Plan**

We found a voicemail plan stood out as one of the most important features. As seen in the graph, people with a voicemail plan are twice as less likely to churn. 

Therefore, we recommend offering voicemail plans to customers who do not have them as part of the incentives used to retain customers. Perhaps when a customer calls the second or third time, SyrianTel can offer them a voicemail plan as a promotion if they don’t currently have one. 

**International Plan**

An international plan was also an important feature. Customers who had an international plan were four times as likely to churn. This is an element SyriaTel should focus on. Perhaps they could consider eliminating this specific plan and offer one reoccurring plan for all. 

**Customer Service calls**

As expected, Customer Service calls were shown to be an important feature. As we see above, customers are five times more likely to churn after the third call. This supports our suggestion of offering an incentive to stay after the second and third call. SyriaTel can offer three weeks free of charge before subscribing for a year or as mentioned above, gift a customer a voicemail plan for three weeks as well. 


**States**

In addition to the states mentioned on our GitHub(code) page, Oregon (OR) should be flagged, as it came out to be an important feature.  Customer Service should be aware of the states that customers are calling from. We recommend exploring the possibility of partnering with other companies. For instance – if a customer from Oregon calls the second time and already has a voicemail plan, one incentive could be to offer a gift from another vendor such as Uber EATS – e.g. a $10 credit to order food which might incentivize the client to stay. 


**Next Step**

* We would like to gather more data on the specific dates of churning. Ideally, we would be able to look at an individual account and learn the dates of onboarding and subsequently leaving. 

* We will look closely into the customer satisfaction. We would like to evaluate the cost of offering a survey once a Customer Service call is complete. We will monitor how long a customer waited before his request was answered.

* We will examine whether a flat fee per month would be more cost-effective than the current method of charge per minute.

* Additionally, we will consider using a different vendor or temporally partnering to offer incentives and promotions when a customer seems dissatisfied. This may increase satisfaction and reduce churning.

* Ultimately, we will implement the new features to see whether churning was reduced and calculate the cost of retaining the customers. 


![](https://i.imgur.com/wgQyKKx.jpg)









