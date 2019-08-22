---
layout: post
title:      "Predicting House Prices in King County using Multi-Regression Model"
date:       2019-08-21 22:28:48 -0400
permalink:  predicting_house_prices_in_king_county_using_multi-regression_model
---

## Data Visualization







![](https://i.imgur.com/b42dnKT.jpg)


 
 
Imagine you and your friend have decided to partner up to launch a local real estate firm. Your goal is to purchase houses and then sell them to make a profit.
 
As part of the research, we would like to learn what would be the most profitable elements when selling a house in the King County area.  

We aim to identify the most important features to consider when selling a house. The results will help us and the investors to know which houses to buy or if there is anything they can improve in the house while selling it.
 
In this blog post, I will focus on how we utilize visualization in the project. We have substantial data and we will use the OSEMiN approach to tackle our goal. 
 

### NaNs

As soon as we upload our data, we start “scrubbing it”. One of the first elements we look at is the NaN values. First, we look at it from a numerical perspective and sum of the NaN values for each variable.
 
```
In: df.isnull().sum()
```

 ```

Out:
 
id              	0
date            	0
price           	0
bedrooms        	0
bathrooms       	0
sqft_living     	0
sqft_lot        	0
floors          	0
waterfront   	2376
view           	63
condition       	0
grade           	0
sqft_above      	0
sqft_basement   	0
yr_built        	0
yr_renovated 	3842
zipcode         	0
lat             	0
long            	0
sqft_living15   	0
sqft_lot15      	0
dtype: int64

```


 
 
When dealing with large data, it can be helpful to visualize the NaN values, especially when there are a lot of them. It is not always easy to grasp it numerically.

For instance, initially when we looked into the NaN values in the year_renovated column numerically using  df.isnull().sum(), it seemed that 3,842(number of NAN values)/21,597 (number of data points per column) =18% is not that significant. Supposedly 17,755 data points per column can appear as enough data. 

However, looking at the heatmap below strongly suggests that there are way too many NaNs in year_renovated. Therefore the best way to handle these NaN values is to remove the year_renovated values altogether or else we will lose other important data from the other columns.   
 
 ```

sns.heatmap(df.isnull(), cbar=False)
plt.title("NaNs")
plt.xlabel('Culomns')
plt.ylabel('Row')
plt.show()
```

 
![](https://i.imgur.com/ALxa3rM.png)
 
 
 
 Additionally,  the waterfront column appears to have a large number of NaN values but in this case, since we presume that this variable will have a significant impact on the house pricing, we will not remove it but rather replace the missing numerical value with the data that appears the most in that column.
 
### Zipcode & Other correlations
 
We look closely at the zip code data to check if there is a pattern, such as if there are certain zip code areas where prices vary tremendously from one another. What we found looking at the graph is that on average there isn’t much difference. We would have had to divide the zip codes into a range of prices but we didn’t think this would be a good variable to examine and focus on given that other variables suggested a greater degree of correlation to housing prices. 
 
```
GB_ZC.plot(kind='bar',x='zipcode',y='price',figsize=(20,20))
plt.savefig('output.png')
plt.title('Price accorsing to Zipcode' )
plt.xlabel('Zipcode')
plt.ylabel('Prices')
plt.show()

```

 
 
 
![](https://i.imgur.com/VcIh0BY.png)





```
sns.set(style="white")

corr = df_final.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

```



![](https://i.imgur.com/ULF3Odq.png)

 ### Waterfront 

Waterfront variable is an interesting one and required us to dig deeper to understand the relationship of waterfront view to the property prices. When dealing with categorical data, especially when it is binary, it is helpful to use Box Plot graph.
 

```

from scipy import stats, linalg
 
fig, ax = plt.subplots(figsize=(12,4))
 
sns.boxplot(y = df['waterfront'], x = df['price'],width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
plt.title('With or Without Waterfront')
plt.show()
 
r, p = stats.pointbiserialr(df['waterfront'], df['price'])
print ('point biserial correlation r is %s with p = %s' %(r,p))
 
 ```

 
 ![](https://i.imgur.com/M26pKLH.png)
 
 
We conclude from this picture the median is much higher for waterfront houses ($1.7million) versus non-waterfront ($0.5 million) and the range of prices is wider with a house that has a waterfront view. (Waterfront view: $0.5 – $5.5 Million, Non-waterfront view:$0.2- $1.3 Million)
 
 
 ###Exploring the data 
 
We are exploring the data and specifically looking into the linear relation of the independent variable to the dependent ones. It is helpful to extrapolate each variable and detect the relationship it has with the target variable. (e.g. Prices of houses)

Let’s examine Sqft_Living.  The code below takes the indicated columns and builds a join graph of density and scatter plot vs. price. We show the sqft_living example in this post. 
 
 ```


for column in [ 'sqft_living','bathrooms', 'bedrooms', 'sqft_lot', 'sqft_above']:
    sns.jointplot(x=column, y="price",
                  data=df, 
                  kind='reg', 
                  label=column,
                  joint_kws={'line_kws':{'color':'green'}})
    plt.legend()
    plt.show()
	```
	

![](https://i.imgur.com/QmRcGT4.pnghttp)


The a joint graph for which shows a fairly strong relationship between the housing price and sqft of living. 

We can see the scatter plot in the middle as well as the linear line with an upward slope. 

Additionally, the density of the variables is presented on each side of the graph. The density suggests that the data is normally distributed.
 
This graph above shows that there is a linear relationship between the price and size of the sqft_ of_living.



### Important confirmation


Lastly, we need to confirm that our residuals are normally distributed or else our P-values would be incorrect.

Following a split train test, we can use this simple code to graph the residuals. 

```

sns.distplot(train_residuals)
plt.title('Residulas' )
plt.xlabel('X')
plt.ylabel('density')
plt.show()

```


![](https://i.imgur.com/iEtIJr2.png)


**This concludes our visual components of the project:** ***predicting house prices in King County using the Multi- Regression Model.***

