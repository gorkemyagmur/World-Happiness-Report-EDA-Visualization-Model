#!/usr/bin/env python
# coding: utf-8

# # Context
# The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.

# # Content
# The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale. The scores are from nationally representative samples for the years 2013-2016 and use the Gallup weights to make the estimates representative. The columns following the happiness score estimate the extent to which each of six factors – economic production, social support, life expectancy, freedom, absence of corruption, and generosity – contribute to making life evaluations higher in each country than they are in Dystopia, a hypothetical country that has values equal to the world’s lowest national averages for each of the six factors. They have no impact on the total score reported for each country, but they do explain why some countries rank higher than others.

# # The Focus of The Study
# 
# The focus of the study is to observe the differences between countries with the highest and lowest happiness scores. In my observations, I observed that the happiness scores of countries depend on different criteria in the happiest and unhappiest countries. Let's start the analysis.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
from pandas_profiling import ProfileReport


# In[2]:


df1 = pd.read_csv('2015.csv')
df2 = pd.read_csv('2016.csv')
df3 = pd.read_csv('2017.csv')
df4 = pd.read_csv('2018.csv')
df5 = pd.read_csv('2019.csv')


# In[3]:


#We must add Year info to our datasets to make more clear.
def year(data,i):
    Year = []
    for column in range(0,len(data)):
        Year.append(i)
        data['Year'] = pd.DataFrame(Year)

#This function changes name of the columns        
def rename_column(data,column,new_name):
    data[new_name] = data[column].rename(new_name)
    data.drop(column,axis = 1,inplace = True)
#This function drops columns.
def dropper(data,column):
    data.drop(column,axis = 1,inplace = True)


# In[4]:


year(df1,2015)
year(df2,2016)
year(df3,2017)
year(df4,2018)
year(df5,2019)


# In[5]:


df_2015 = df1
df_2016 = df2
df_2017 = df3
df_2018 = df4
df_2019 = df5


# In[6]:


rename_column(df_2015,'Happiness Score','Score')
rename_column(df_2016,'Happiness Score','Score')
rename_column(df_2017,'Happiness.Score','Score')


# We should drop ranking columns. We can reach that information by Happiness Score column.

# In[7]:


dropper(df_2015,'Happiness Rank')
dropper(df_2016,'Happiness Rank')
dropper(df_2017,'Happiness.Rank')
dropper(df_2018,'Overall rank')
dropper(df_2019,'Overall rank')


# We want to see all columns informations.

# In[8]:


print(df_2015.info())


# In[9]:


print(df_2016.info())


# In[10]:


print(df_2017.info())


# In[11]:


print(df_2018.info())


# In[12]:


print(df_2019.info())


# We need some visualizations to understand our datasets better.

# We have 5 different datasets and we need to look at the same values year by year to understand differences between values. So we need a couple of functions.

# In[13]:


factors_of_2015_all = []
def heatmap_corr(data,corr_with_score):
    plt.figure(figsize = (12,6))
    sns.heatmap(data.corr(),annot = True,cmap = 'icefire_r')
    plt.show()            
    print(""" These are the correlations for happiness score of all countries in {}\n      
{}
        
""".format(data['Year'].iloc[1],pd.DataFrame(data.corr()[corr_with_score].sort_values(ascending = False)[1:])))
    
def heatmap_corr_head(data,corr_with_score):
    plt.figure(figsize = (12,6))
    sns.heatmap(data.head().corr(),annot = True,cmap = 'Set3')
    plt.show()
    print(""" These are the correlations for countries with highest happiness score in {}\n      
{}
        
""".format(data['Year'].iloc[1],pd.DataFrame(data.head().corr()[corr_with_score].sort_values(ascending = False)[1:])))

def heatmap_corr_tail(data,corr_with_score):
    plt.figure(figsize = (12,6))
    sns.heatmap(data.tail().corr(),annot = True,cmap = 'ocean_r')
    plt.show()
    print(""" These are the correlations for countries with lowest happiness score {}\n
    {}
    """.format(data['Year'].iloc[1],pd.DataFrame(data.tail().corr()[corr_with_score].sort_values(ascending = False)[1:])))


# In[14]:


heatmap_corr(df_2015,'Score')
heatmap_corr_head(df_2015,'Score')
heatmap_corr_tail(df_2015,'Score')


# In[15]:


heatmap_corr(df_2016,'Score')
heatmap_corr_head(df_2016,'Score')
heatmap_corr_tail(df_2016,'Score')


# In[16]:


heatmap_corr(df_2017,'Score')
heatmap_corr_head(df_2017,'Score')
heatmap_corr_tail(df_2017,'Score')


# In[17]:


heatmap_corr(df_2018,'Score')
heatmap_corr_head(df_2018,'Score')
heatmap_corr_tail(df_2018,'Score')


# In[18]:


heatmap_corr(df_2019,'Score')
heatmap_corr_head(df_2019,'Score')
heatmap_corr_tail(df_2019,'Score')


# # Some Conclusions

#     According to correlations between happiness scores and other values, we can notice that people who living in happier countries and people who living in sadder countries have different criterias for happiness. 
#     In 2015 for people in happier countries, most important criteria is their family. Also in 2015 the most important criteria for  saddest countries is freedom. 
#     In 2016 for people in happier countries, most important criteria is confidence intervals, for poeople saddest countries the most important criterias are also confidence intervals and freedom just like 2015.
#     In 2017 for people in happier countries, most important criteria is freedom, for poeople saddest countries the most important criterias are GDP and Healty Life Expectancy.
#     In 2018 for people in happier countries, most important criteria is freedom, for poeople saddest countries the most important criteria is GDP.
#     In 2019 for people in happier countries, most important criteria is Perceptions of corruption, for poeople saddest countries the most important criteria is Freedom to make life choices.

# In[19]:


def lmplot(column_x,column_y,data):
    print("""
    
    Correlation between {} and {} is {}
    
    """.format(column_x,column_y,data.corr()[column_x][column_y]))
    
    sns.lmplot(x = column_x, y = column_y,data = data,palette = 'Paired',hue ='Country')
    
    plt.title(data['Year'].iloc[1])
    
    plt.show()
   


# # We got a couple of datasets but in this study I will focus on the differences of countries in 2015, different top or bottom countries between 2015-2016 and 2016-2017, also I will chech whether there is a country that has a dramatic change.
# Our goal is analyzing differences between happiest and saddest countries. We will compare them in the plots.

# In[20]:


lmplot('Score','Family',df_2015.head(10))
lmplot('Score','Family',df_2015.tail(10))


# In[21]:


lmplot('Score','Economy (GDP per Capita)',df_2015.head(10))
lmplot('Score','Economy (GDP per Capita)',df_2015.tail(10))


# In[22]:


lmplot('Score','Freedom',df_2015.head(10))
lmplot('Score','Freedom',df_2015.tail(10))


# # Top 10 Country - Health(Expentacy)

# In[23]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Health (Life Expectancy)', data = df_2015.head(10))
plt.xticks(rotation = 90)
plt.show()


# # Bottom 10 Country - Health(Expentacy)

# In[24]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Trust (Government Corruption)', data = df_2015.tail(10),palette = 'inferno_r')
plt.xticks(rotation = 90)
plt.show()


# # Top 10 Country - Trust (Government Corruption)

# In[25]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Trust (Government Corruption)', data = df_2015.head(10))
plt.xticks(rotation = 90)
plt.show()


# # Bottom 10 Country - Trust (Government Corruption)

# In[26]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Trust (Government Corruption)', data = df_2015.tail(10),palette = 'inferno_r')
plt.xticks(rotation = 90)
plt.show()


# In[27]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Score', data = df_2015.head(10),palette = 'inferno_r')
plt.xticks(rotation = 90)
plt.show()


# In[28]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Score', data = df_2016.head(10),palette = 'inferno_r')
plt.xticks(rotation = 90)
plt.show()


# In[29]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country', y = 'Score', data = df_2017.head(10),palette = 'inferno_r')
plt.xticks(rotation = 90)
plt.show()


# # Northern European countries seem to be in the top positions every year.

# # Now let's look for a dramatic change.

# In[30]:


rename_column(df_2019,'Country or region','Country')


# In[31]:


df_2015_country_score = pd.DataFrame(df_2015[['Score','Country']])
df_2019_country_score = pd.DataFrame(df_2019[['Score','Country']])


# In[32]:


united_df_2015_2019 = df_2015_country_score.merge(df_2019_country_score,how = 'inner',on = 'Country')


# In[33]:


rename_column(united_df_2015_2019,'Score_x','Score_2015')
rename_column(united_df_2015_2019,'Score_y','Score_2019')


# In[34]:


united_df_2015_2019['Difference_Between_Scores_In_4_Years'] = united_df_2015_2019['Score_2019']-united_df_2015_2019['Score_2015']


# # The countries that have most dramatic rise in 4 years.

# In[35]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country',y = 'Difference_Between_Scores_In_4_Years',data = united_df_2015_2019.sort_values(ascending = False,by = 'Difference_Between_Scores_In_4_Years').head(10),palette = 'gist_rainbow')
plt.xticks(rotation = 90)
plt.show()


# In[36]:


united_df_2015_2019.sort_values(ascending = False,by = 'Difference_Between_Scores_In_4_Years').head(10)


# # The countries that have most dramatic fall in 4 years.

# In[37]:


plt.figure(figsize=(25, 10))
sns.barplot(x = 'Country',y = 'Difference_Between_Scores_In_4_Years',data = united_df_2015_2019.sort_values(ascending = False,by = 'Difference_Between_Scores_In_4_Years').tail(10),palette = 'flare_r')
plt.xticks(rotation = 90)
plt.show()


# In[38]:


united_df_2015_2019.sort_values(ascending = False,by = 'Difference_Between_Scores_In_4_Years').tail(10)


# # Now let's build the model

# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# In[40]:


df_2015['Region'].value_counts()


# In[41]:


le = LabelEncoder()
df_2015['Region'] = le.fit_transform(df_2015['Region'])


# In[42]:


X = df_2015.drop(['Country','Year','Score'],axis = 1)
y = df_2015['Score']


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[44]:


scaler = MinMaxScaler()


# In[45]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[46]:


from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[47]:


xgb = XGBRegressor()
lr = LinearRegression()
tre = DecisionTreeRegressor()
forest = RandomForestRegressor(n_estimators=10)
grad = GradientBoostingRegressor()
ada = AdaBoostRegressor()
kn = KNeighborsRegressor()


# In[48]:


models = [xgb,lr,tre,forest,grad,ada,kn]


# We got a few models so we put all of them in a loop to work fast.

# In[49]:


for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test,y_test)
    print(model)
    print(score)
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('R2 Score:', r2_score(y_test, y_pred))
    print('---------------------------------------------------')


# We can say that LinearRegression method has slightly better score. Now let's check.

# In[50]:


lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
lr_score = lr.score(X_test,y_test)


# In[51]:


plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='red')
plt.plot(y_test, y_test, color='green')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('Linear Regression')
plt.show()


# Finally we try the model with real data values. Let's consider Turkey.

# In[52]:


df_2015[df_2015['Country']=='Turkey']


# In[53]:


input_data = [4,0.03864,1.06098,0.94632,0.73172,0.22815,0.15746,0.12253,2.08528]
input_data_as_array = np.asarray(input_data)
input_data_as_array_reshaped = input_data_as_array.reshape(1,-1)
input_ = scaler.transform(input_data_as_array_reshaped)
lr_fit = lr.fit(X_train,y_train)
pred = lr_fit.predict(input_)


# In[54]:


print("""
Our real data score for Turkey is {}

Predict of our model is {}

""".format(df_2015[df_2015['Country']=='Turkey']['Score'].values,pred))


# # It looks like our model is successfull

# # Let's try cross validation to reaching best possible score on our model.

# In[55]:


from sklearn.model_selection import cross_val_score


# In[56]:


score_on_train = cross_val_score(lr_fit,X_train,y_train,scoring = 'r2',cv = 5)
score_on_test = cross_val_score(lr_fit,X_test,y_test,scoring = 'r2',cv = 5)


# In[57]:


score_on_test


# In[58]:


score_on_train


# # With Cross Validation, out score became nearly perfect.

# # Conclusions

# As a result, different countries have different criteria for happiness score. In this study, we saw that the happiest countries and the relatively unhappiest countries have different lifestyles and happiness criteria. In the correlations made for each year, we saw that the happiness criteria changed from year by year. We can say that the samples that decide on the criteria of the research change because the happiness criteria all over the world cannot be changed from year to year. The sample must be selected and stay constant in order to make better analyzes in the future.

# In[ ]:




