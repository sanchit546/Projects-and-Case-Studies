#!/usr/bin/env python
# coding: utf-8

# # IMPORTING THE NECESSARY MODULES

# In[181]:


#importing all the important libraries like numpy. pandas, matlplolib, and warnings to keep notebook clean

import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[182]:


# to suppress warnings

import warnings   
warnings.filterwarnings("ignore")


# In[183]:


#notebook setting to display all the rowns and columns to have better clearity on the data.

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


# # Dataset 1 - application_data.csv

# In[184]:


# importing application_data.csv

application_data = pd.read_csv("application_data.csv")


# In[185]:


application_data.head()


# # UNDERSTANDING THE DATASET

# In[186]:


application_data.shape


# In[187]:


application_data.info("all")


# In[188]:


# Checking the numeric variables of the dataframes

application_data.describe()


# **There are 122 columns and 307511 rows. There columns having negative, postive values which includes days. There are columns with very high values, columns related to Amount(Price).**

# In[189]:


#checking the columns of the dataframe

application_data.columns


# # Data Cleaning & Manipulation

# In[190]:


#checking how many null values are present in each of the columns

#creating a function to find null values for the dataframe
def null_values(df):
    output = round(df.isnull().sum()/len(df.index)*100,2)
    return output


# In[191]:


# Null values of all columns

null_values(application_data)


# In[192]:


#Null Values in Descending Order
null_values(application_data).sort_values(ascending=False)


# In[193]:


# Storing null values of columns in a variable named Null column
Null_column = null_values(application_data)


# In[194]:


#Finding out columns with only null values

Null_column = Null_column[Null_column>0]
Null_column


# In[195]:


Null_column.count


# In[196]:


len(Null_column)


# **There are 64 columns which have null values**

# In[197]:


Null_column.index


# In[198]:


#Visualising Null values in columns in bar graph

plt.figure(figsize = (10,3), dpi=100)
Null_column.plot(kind = "bar")
plt.title("Null values in columns")
plt.xlabel('Percentage of Null values')
plt.show()


# In[199]:


#creating a variable Null_column_40 for storing null columns having missing values more than 40%

Null_column_40 = null_values(application_data)[null_values(application_data)>40]


# In[200]:


Null_column_40


# In[201]:


print("Number of columns having missing values more than 40% :",len(Null_column_40))


# In[202]:


Null_column_40.index


# In[203]:


# Now lets drop all the columns having missing values more than 40% that is 49 columns

application_data.drop(columns = Null_column_40.index, inplace = True)


# In[204]:


application_data.shape


# **After after dropping 49 columns we are left with 73 columns**

# In[205]:


null_values(application_data)


# **After after dropping 49 columns with null values more than 40% we are left with 73 columns**

# In[206]:


#Columns with null values <15%
Null_column_15 = Null_column[Null_column<15]
print(Null_column_15)


# In[207]:


print("Number of columns with null value < 15% :", len(Null_column_15.index))


# **There are 13 columns which have null values less than 15%**

# In[208]:


Null_column_15.index


# In[209]:


application_data[Null_column_15.index].nunique().sort_values(ascending=False)


# **From the above we can see that first two (EXT_SOURCE_2, AMT_GOODS_PRICE) are continous variables and remaining are catagorical variables**

# In[210]:


application_data


# **Now We will Draw boxplot of columns with highest unique values and lowest unique values**

# In[211]:


#column with highest unique values
plt.figure(figsize=(5,2))
sns.boxplot(application_data['EXT_SOURCE_2'])
plt.show()


# In[212]:


plt.figure(figsize=(5,2))
sns.boxplot(application_data['AMT_GOODS_PRICE'])
plt.show()


# In[213]:


plt.figure(figsize=(5,2))
sns.boxplot(application_data['OBS_30_CNT_SOCIAL_CIRCLE'])
plt.show()


# In[214]:


#Column with lowest unique values
plt.figure(figsize=(5,2))
sns.boxplot(application_data['AMT_REQ_CREDIT_BUREAU_HOUR'])
plt.show()


# **For 'EXT_SOURCE_2', column with most unique values there no outliers present. So data is rightly present. For 'AMT_GOODS_PRICE','AMT_REQ_CREDIT_BUREAU_HOUR','OBS_30_CNT_SOCIAL_CIRCLE' outliers are present in the data.**
# 
# **This is just for displaying we will deal will Outliers further in the analysis**
# 

# In[215]:


#Correlation between EXT_SOURCE_2,AMT_GOODS_PRICE, TARGET
cont = ["EXT_SOURCE_2","AMT_GOODS_PRICE"]
plt.figure(figsize= [5,3])

sns.heatmap(application_data[cont+["TARGET"]].corr(), cmap="Reds",annot=True)

plt.title("Correlation between EXT_SOURCE_2,AMT_GOODS_PRICE, TARGET", fontdict={"fontsize":15}, pad=25)
plt.show()


# In[216]:


#Correlation between AMT_INCOME_TOTAL,AMT_GOODS_PRICE, TARGET
cont = ["AMT_INCOME_TOTAL","AMT_GOODS_PRICE"]
plt.figure(figsize= [5,3])

sns.heatmap(application_data[cont+["TARGET"]].corr(), cmap="Reds",annot=True)

plt.title("Correlation between AMT_INCOME_TOTAL,AMT_GOODS_PRICE, TARGET", fontdict={"fontsize":15}, pad=25)
plt.show()


# **There seems to be no linear correlation**
# 
# **Also we are aware correation doesn't cause causation**
# 
# **We will analyse correlation deeply and heatmaps further in the analysis**

# In[217]:


application_data.head()


# In[218]:


# adding all flags coloumns in variable "flag_columns"

flag_columns = [col for col in application_data.columns if "FLAG" in col]

flag_columns  # Viewing all FLAG columns


# In[219]:


Flag_dataframe = application_data[flag_columns+["TARGET"]]


# In[220]:


Flag_dataframe["TARGET"] = Flag_dataframe["TARGET"].replace({1:"Defaulter", 0:"Repayer"})


# In[221]:


for i in Flag_dataframe:
    if i!= "TARGET":
        Flag_dataframe[i] = Flag_dataframe[i].replace({1:"Y", 0:"N"})


# In[222]:


Flag_dataframe.head()


# In[223]:


import itertools # using itertools for efficient looping plotting subplots

# Plotting all the graph to find the relation and evaluting for dropping such columns

plt.figure(figsize = [24,28])

for i,j in itertools.zip_longest(flag_columns,range(len(flag_columns))):
    plt.subplot(7,4,j+1)
    ax = sns.countplot(Flag_dataframe[i], hue = Flag_dataframe["TARGET"], palette = ["r","y"])
    #plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# **Columns (FLAG_OWN_REALTY, FLAG_MOBIL ,FLAG_EMP_PHONE, FLAG_CONT_MOBILE, FLAG_DOCUMENT_3) have more repayers than defaulter and from these keeping FLAG_DOCUMENT_3,FLAG_OWN_REALTY, FLAG_MOBIL more sense thus we can include these columns and drop all other FLAG columns for furhter analysis.**

# In[224]:


# removing required columns from "flag_df" such that we can remove the irrelevent columns from "application_data" dataset.

Flag_dataframe.drop(["TARGET","FLAG_OWN_REALTY","FLAG_MOBIL","FLAG_DOCUMENT_3"], axis=1 , inplace = True)


# In[225]:


len(Flag_dataframe.columns)


# In[226]:


# dropping the columns of "Flag_df" dataframe that is removing more 25 columns from "appliation_data" dataframe

application_data.drop(Flag_dataframe.columns, axis=1, inplace= True)


# In[227]:


application_data.shape


# **After removing uneccsarry, irrelevent and missing columns. We are left with 46 columns**

# # Imputing Values and Fixing Necessary Irregularities

# In[228]:


application_data


# **DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, These columns have values in negative so we will convert them to positive**

# In[229]:


#Age/Days columns are in -ve which needs to be converted to +ve value
application_data.DAYS_BIRTH = application_data.DAYS_BIRTH.abs()
application_data.DAYS_EMPLOYED = application_data.DAYS_EMPLOYED.abs()
application_data.DAYS_REGISTRATION = application_data.DAYS_REGISTRATION.abs()
application_data.DAYS_ID_PUBLISH = application_data.DAYS_ID_PUBLISH.abs()


# In[230]:


application_data


# **Values have been converted to positive**

# **Next, In the CODE_GENDER Column we have XNA values, So we will Fix them**

# In[231]:


# Imputing the value'XNA' which means not available for the column 'CODE_GENDER'

application_data.CODE_GENDER.value_counts()


# **XNA values are very low and Female is the majority. So lets replace XNA with gender 'F'**

# In[232]:


application_data.loc[application_data.CODE_GENDER == 'XNA', 'CODE_GENDER'] = 'F'


# In[233]:


application_data.CODE_GENDER.value_counts()


# In[234]:


# checking the CODE_GENDER 

application_data.CODE_GENDER.head(10)


# **Only M and F values are present now and XNA values have been replaced**

# In[235]:


#null_values in application_data
null_values(application_data)


# In[236]:


#Storing all the columns with null values more than 7% in Null_column_1
Null_column_1 = null_values(application_data)[null_values(application_data)>1]


# In[237]:


Null_column_1


# In[238]:


len(Null_column_1)


# **There are 8 columns with missing values more than 1%**

# In[239]:


#We can see that 7 columns except Occupation_type have numerical values so first we will fix them
application_data[["AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK",
"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR","EXT_SOURCE_3"]].describe()


# In[240]:


#imputing median value in missing values of "AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR","EXT_SOURCE_3"

application_data.fillna(application_data[["AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK",
"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR","EXT_SOURCE_3"]].median(),inplace = True)


# In[241]:


null_values(application_data).head(10)


# **We have achieved negligible null values in these columns and replaced the missing values with Median**

# In[242]:


#For the occupation_type Column which has 31.35% null values we should replace the missing values
# imputing null values with "Missing"

application_data["OCCUPATION_TYPE"] = application_data["OCCUPATION_TYPE"].fillna("Missing") 


# In[243]:


application_data["OCCUPATION_TYPE"].isnull().sum()


# **Now we have zero null values in Occupation Column**

# In[244]:


application_data


# In[245]:


# Plotting a percentage graph having each category of "OCCUPATION_TYPE"

plt.figure(figsize = [12,7])
(application_data["OCCUPATION_TYPE"].value_counts()).plot.barh(color= "green",width = .9)
plt.title("Percentage of Type of Occupations ", fontdict={"fontsize":18}, pad =20)
plt.show()


# **We can see that most applicants are missing one after that laborers are the second highest occupation type**

# In[246]:


application_data.shape


# # IDENTIFYING OUTLIERS

# In[247]:


application_data.describe()


# In[248]:


#storing "CNT_CHILDREN","AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE","DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION" 
outlier = ["CNT_CHILDREN","AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
               "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION"]


# In[249]:


#plotting boxplot for displaying outliers in outlier variable in which we stored "CNT_CHILDREN","AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE","DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION" in outlier

plt.figure(figsize=[15,25])
for i,j in itertools.zip_longest(outlier, range(len(outlier))):
    plt.subplot(4,2,j+1)
    sns.boxplot(y = application_data[i], orient = "h", color = "red")
    #plt.yticks(fontsize=6)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# **1st quartile is missing for CNT_CHILDREN which means most of the data are present in the 1st quartile**
# 
# **In AMT_INCOME_TOTAL only single high value data point is present as outlier.**
# 
# **AMT_CREDIT has little bit more outliers.**
# 
# **1st quartiles and 3rd quartile for AMT_ANNUITY is moved towards first quartile.**
# 
# **1st quartiles and 3rd quartile for DAYS_EMPLOYED is stays towrads first quartile.**
# 

# **It can be seen that in current application data**
# 
# **AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE,CNT_CHILDREN have some number of outliers.**
# 
# **AMT_INCOME_TOTAL has huge number of outliers which indicate that few of the loan applicants have high income when compared to the others.**
# 
# **DAYS_BIRTH has no outliers which means the data available is reliable.**
# 
# **DAYS_EMPLOYED has outlier values around 350000(days) which is around 958 years which is impossible and hence this has to be incorrect entry.**

# # DATA IMBALANCE

# In[250]:


application_data


# In[251]:


#Creating variable Target_0 for non defaulters and #Target_1 for defaulters
Target_0 = application_data.loc[application_data.TARGET == 0]
Target_1 = application_data.loc[application_data.TARGET == 1]


# In[252]:


application_data.TARGET.head()


# In[253]:



Imbalance = round(len(Target_0)/len(Target_1),2)

print('Imbalance Ratio:', Imbalance)


# **11:39 is the imbalance ratio**

# In[254]:


#Plotting bar graph for comparison between Target_0(Non defaulters) and Target_1(defaulters)
plt.figure(figsize= [14,5])
sns.barplot(y=["Target_0","Target_1"], x = application_data["TARGET"].value_counts(), palette = ["blue","y"],orient="h")
plt.ylabel("Loan Repayment Status",fontdict = {"fontsize":15})
plt.xlabel("Count",fontdict = {"fontsize":15})
plt.title("Imbalance Plotting (Non Defaulters Vs Defaulters)", fontdict = {"fontsize":20}, pad = 20)
plt.show()


# In[255]:


#Ratio of imbalance percentage with respect to defaulter and repayer is given below 
Non_defaulters = round((application_data["TARGET"].value_counts()[0]/len(application_data)* 100),2)
print("Target_0 Percentage is {}%".format(Non_defaulters))
Defaluters = round((application_data["TARGET"].value_counts()[1]/len(application_data)* 100),2)
print("Target_1 Percentage is {}%".format(Defaluters))
print("Imbalance Ratio with respect to Non Defaulters and Defaulters is given: {0:.2f}/1 (approx)".format(Non_defaulters/Defaluters))


# **Non Defaulters Percentage is 91.93%**
# 
# **Defaulters Percentage is 8.07%**
# 
# **Imbalance Ratio with respect to Non Defaulters and Defaulters is given: 11.39/1 (approx)**
# 
# **This shows that practically Non Defaulters are much more higher as compared to Defaulters. Which is in real scenario also true, because if the number Of Defaulters will be high than the bank will suffer heavy losses.**

# # IDENTIFYING NUMERICAL AND CATEGORICAL COLUMNS

# In[256]:


#storing columns with float values in category_col
category_col=application_data.select_dtypes(include=['object']).columns


# In[257]:


category_col


# **These are categorical colums and we will perform categorical analysis on them**

# In[258]:


#storing values with int and float into numerical_col
numerical_col=application_data.select_dtypes(include=['int64','float64'])


# **These columns are Numerical and we will perform numerical analysis on them**

# In[259]:


numerical_col


# # UNIVARIATE ANALYSIS

# In[260]:


#Univariate analysis of 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE'
AMT= application_data[[ 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']]

fig = plt.figure(figsize=(16,12))

for i in enumerate(AMT):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(Target_0[i[1]], hist=False,label ="Non Defaulter")
    sns.distplot(Target_1[i[1]], hist=False, label ="Defaulter")
    plt.title(i[1], fontdict={'fontsize' : 16, 'fontweight' : 6})
    plt.legend()


# **Most no of loans are given for goods price below 10 lakhs**
# 
# **Most people pay annuity below 50K for the credit loan**
# 
# **Credit amount of the loan is mostly less then 10 lakhs**
# 
# **The repayers and defaulters distribution overlap in all the plots and hence we cannot use any of these variables in isolation to make a decision**

# In[261]:


#distribution of AMT_INCOME_TOTAL
appli_ait=application_data[['AMT_INCOME_TOTAL']]

#create bins 

bins=[0,10000,20000,30000,40000,50000,60000,70000,80000,90000]

#plotting of histogram

appli_ait.hist(bins=bins)
plt.show()


# In[262]:


appli_1_amt_income_total=Target_1[['AMT_INCOME_TOTAL']]


# In[263]:


appli_0_amt_income_total=Target_0[['AMT_INCOME_TOTAL']]


# In[264]:


min=appli_1_amt_income_total.describe().min()
max=appli_1_amt_income_total.describe().max()


# In[265]:


min


# In[266]:


max


# In[267]:


bins0=[i for i in range(0,400000,10000)]
range0=[min['AMT_INCOME_TOTAL'],max['AMT_INCOME_TOTAL']]


# In[268]:


range0


# In[269]:


#plotting of histogram of customer with difficulty 
plt.figure(figsize=(25,12))
appli_1_amt_income_total.hist(bins=bins0,range=range0)
plt.title('Customer with difficulty')
plt.show()
range1=[min['AMT_INCOME_TOTAL'],max['AMT_INCOME_TOTAL']]
range1=[min['AMT_INCOME_TOTAL'],max['AMT_INCOME_TOTAL']]


# **Most of the loan defaults is for clients whose income between :1100000-1500000**

# In[270]:


Target_0.describe()


# In[271]:


Target_1.describe()


# In[272]:


appli_0_amt_credit=Target_0[['AMT_CREDIT']]
appli_1_amt_credit=Target_1[['AMT_CREDIT']]


# In[273]:


min_ac_0=appli_0_amt_credit.describe().min()
max_ac_0=appli_0_amt_credit.describe().max()


# In[274]:


min_ac_1=appli_1_amt_credit.describe().min()
max_ac_1=appli_1_amt_credit.describe().max()


# In[275]:


bin_ac = [i for i in range(0,400000,10000)]


# In[276]:


#Numerical analysis for AMT_CREDIT for customers with difficulty and no difficulty
plt.figure(figsize=(30,15))
range_ac_0=[max_ac_0['AMT_CREDIT'],min_ac_0['AMT_CREDIT']]
range_ac_1=[max_ac_1['AMT_CREDIT'],min_ac_1['AMT_CREDIT']]
appli_0_amt_credit.hist(bins=bin_ac,range=range_ac_0)
plt.title('Customer with no difficulty')
appli_1_amt_credit.hist(bins=bin_ac,range=range_ac_1)
plt.title('Customer with difficulty')
plt.show()


# **Most of loan default is for client whose income in between 260000-290000**

# In[277]:


#Numerical analysis for AMT_ANNUITY for customers with difficulty and no difficulty
appli_0_AMT_ANNUITY=Target_0[['AMT_ANNUITY']]
appli_1_AMT_ANNUITY=Target_1[['AMT_ANNUITY']]
min_aa=appli_0_AMT_ANNUITY.describe().min()
max_aa=appli_0_AMT_ANNUITY.describe().max()

min_aa=appli_1_AMT_ANNUITY.describe().min()
max_aa=appli_1_AMT_ANNUITY.describe().max()
bin_aa=[i for i in range(0,100000,2500)]
#plotting of data
plt.figure(figsize=(30,15))
range_aa=[max_aa['AMT_ANNUITY'],min_aa['AMT_ANNUITY']]
range_aa=[max_aa['AMT_ANNUITY'],min_aa['AMT_ANNUITY']]
appli_0_AMT_ANNUITY.hist(bins=bin_aa,range=range_aa)
plt.title('Customer with no difficulty')
appli_1_AMT_ANNUITY.hist(bins=bin_aa,range=range_aa)
plt.title('Customer with difficulty')
plt.show()


# In[278]:


# PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR NAME_HOUSING_TYPE
plt.figure(figsize=(16,8))
sns.countplot(Target_0['NAME_HOUSING_TYPE']).set_title("People without Difficulty")
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['NAME_HOUSING_TYPE']).set_title("People with Difficulty")
plt.show()


# **Most of the applicants own real estate**

# In[279]:


# PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR NAME_EDUCATION_TYPE
plt.figure(figsize=(16,8))
sns.countplot(Target_0['NAME_EDUCATION_TYPE']).set_title("People without Difficulty")
plt.xticks(rotation='vertical')
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['NAME_EDUCATION_TYPE']).set_title("People with Difficulty")
plt.xticks(rotation='vertical')
plt.show()


# **People with Academic degree are least likely to default.**
# 

# In[280]:


#PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR CODE_GENDER
plt.figure(figsize=(16,8))
sns.countplot(Target_0['CODE_GENDER']).set_title("People without Difficulty")
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['CODE_GENDER']).set_title("People with Difficulty")
plt.show()


# **Females are less likely to Default**

# In[281]:


#PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR OCCUPATION_TYPE
# PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_0['OCCUPATION_TYPE']).set_title("People without Difficulty")
plt.xticks(rotation='vertical')
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['OCCUPATION_TYPE']).set_title("People with Difficulty")
plt.xticks(rotation='vertical')
plt.show()


# **Most of the loans are taken by Laborers, followed by Sales staff.**
# 
# **IT staff and HR staff are less likely to apply for Loan.**
# 
# 

# In[282]:


#PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR ORGANIZATION_TYPE
# PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_0['ORGANIZATION_TYPE']).set_title("People without Difficulty")
plt.xticks(rotation='vertical')
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['ORGANIZATION_TYPE']).set_title("People with Difficulty")
plt.xticks(rotation='vertical')
plt.show()


# **Self employed people have relative high default rate, to be safer side loan disbursement should be avoided.**
# 
# **Most of the people application for loan are from Business Entity Type 3**
# 
# **For a very high number of applications, Organization type information is unavailable(XNA)**
# 

# In[283]:


#PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR REGION_RATING_CLIENT
#PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_0['REGION_RATING_CLIENT']).set_title("People without Difficulty")
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['REGION_RATING_CLIENT']).set_title("People with Difficulty")
plt.show()


# **Applicants with rating 1 are less likely to default.**

# In[284]:


#PEOPLE WITHOUT DIFFICULTY And WITHOUT DIFFICULTY FOR FLAG_OWN_REALTY
#PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_0['FLAG_OWN_REALTY']).set_title("People without Difficulty")
plt.show()

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(16,8))
sns.countplot(Target_1['FLAG_OWN_REALTY']).set_title("People with Difficulty")
plt.show()


# **The clients who own real estate are more than double of the ones that don't own.**

# # BIVARIATE ANALYSIS

# In[285]:


#bivariate analysis for AMT_INCOME_TOTAL and OCCUPATION_TYPE with respect to Target_0 and Target_1
#people without difficulties
plt.figure(figsize=(15,8))
sns.lineplot(x="AMT_INCOME_TOTAL", y="OCCUPATION_TYPE", data=Target_0)
plt.show()


#people with difficulties
plt.figure(figsize=(15,8))
sns.lineplot(x="AMT_INCOME_TOTAL", y="OCCUPATION_TYPE", data=Target_1)
plt.show()


# In[286]:


#bivariate analysis for AMT_INCOME_TOTAL and CODE_GENDER with respect to Target_0 and Target_1
#people without difficulties
plt.figure(figsize=(15,8))
sns.lineplot(x="AMT_INCOME_TOTAL", y="CODE_GENDER", data=Target_0)
plt.show()


#people with difficulties
plt.figure(figsize=(15,8))
sns.lineplot(x="AMT_INCOME_TOTAL", y="CODE_GENDER", data=Target_1)
plt.show()


# In[287]:


#joint plot b/w AMT_CREDIT AND AMT_INCOME_TOTAL with respect to Target_0 and Target_1
#People without difficulties
plt.figure(figsize=(25,112))
sns.jointplot('AMT_CREDIT','AMT_INCOME_TOTAL',Target_0)
plt.show()
#People with difficulties
plt.figure(figsize=(25,112))
sns.jointplot('AMT_CREDIT','AMT_INCOME_TOTAL',Target_1)
plt.show()


# In[288]:


#joint plot b/w AMT_CREDIT AND AMT_ANNUITY with respect to Target_0 and Target_1
#people without difficulties
plt.figure(figsize=(25,12))
sns.jointplot('AMT_CREDIT','AMT_ANNUITY',Target_0)
plt.show()
#people with difficulties
plt.figure(figsize=(25,12))
sns.jointplot('AMT_CREDIT','AMT_ANNUITY',Target_1)
plt.show()


# # CORRELATION ANALYSIS

# **We Will perform correlation analysis for Target_0(Non Defaulters) and Target_1(Defaulters)**

# In[289]:


application_data.columns


# In[290]:


correlation_columns =  ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY', 
                        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
                        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                        'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 
                        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']


# In[291]:


#Creating dataframe for defaulters and non defaulters for correlation analysis
#Non Defaulters dataframe
Target_0_Correlation= application_data.loc[application_data['TARGET']==0, correlation_columns]

# Defaulters dataframe
Target_1_Correlation = application_data.loc[application_data['TARGET']==1, correlation_columns]


# In[292]:


len(correlation_columns)


# In[293]:


#Finding Top 10 Correlation for Target_0 (Non Defaulters)
corr_Non_defaulter =Target_0_Correlation.corr()
corr_df_non_defaulter = corr_Non_defaulter.where(np.triu(np.ones(corr_Non_defaulter.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df_non_defaulter.columns =['VAR1','VAR2','Correlation']
corr_df_non_defaulter.dropna(subset = ["Correlation"], inplace = True)
corr_df_non_defaulter["Correlation"]=corr_df_non_defaulter["Correlation"].abs() 
corr_df_non_defaulter.sort_values(by='Correlation', ascending=False, inplace=True) 
corr_df_non_defaulter.head(10)


# In[294]:


#Correlation heatmap for non defaulters
Heatmap_Non_defaulters = plt.figure(figsize=(25,20))
ax = sns.heatmap(Target_0_Correlation.corr(), cmap="RdYlGn",annot=True,linewidth =1)


# **Correlating factors amongst Non Defaulters**
# 
# **1. Credit amount is highly correlated with:**
# 
# **Goods Price Amount**
# 
# **Loan Annuity**
# 
# **Total Income**
# 
# **2. We can also see that Non Defaulters have high correlation in number of days employed.**

# In[295]:


#Finding Top 10 correlation for defaulters
corr_Defaulter = Target_1_Correlation.corr()
corr_df_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape),k=1).astype(np.bool))
corr_df_Defaulter = corr_Defaulter.unstack().reset_index()
corr_df_Defaulter.columns =['VAR1','VAR2','Correlation']
corr_df_Defaulter.dropna(subset = ["Correlation"], inplace = True)
corr_df_Defaulter["Correlation"]=corr_df_Defaulter["Correlation"].abs()
corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)
corr_df_Defaulter.head(10)


# In[296]:


#Correlation heatmap for defaulters
Heatmap_defaulters = plt.figure(figsize=(25,20))
ax = sns.heatmap(Target_1_Correlation.corr(), cmap="RdYlGn",annot=True,linewidth =1)


# **Credit amount is highly correlated with good price amount which is same as Non Defaulters.**
# 
# **Loan annuity correlation with credit amount has slightly reduced in defaulters(0.75) when compared to Non Defaulters(0.77)**
# 
# **We can also see that Non Defaulters have high correlation in number of days employed(0.62) when compared to defaulters(0.58).**
# 
# **There is a severe drop in the correlation between total income of the client and the credit amount(0.038) amongst defaulters whereas it is 0.342 among Non Defaulters.**
# 
# **Days_birth and number of children correlation has reduced to 0.259 in defaulters when compared to 0.337 in Non Defaulters.**
# 
# **There is a slight increase in defaulted to observed count in social circle among defaulters(0.264) when compared to Non Defaulters(0.254)**

# # PREVIOUS APPLICATION DATASET

# In[297]:


# importing previous_application.csv

previous_application = pd.read_csv("previous_application.csv")


# In[298]:


previous_application


# In[299]:


previous_application.shape


# In[300]:


previous_application.info('all')


# In[301]:


previous_application.columns


# In[302]:


previous_application.describe()


# **There are 1670214 rows and 37 columns in the previous_application dataset**

# # DATA CLEANING

# In[303]:


#checking null values in the previous application dataset
null_values(previous_application)


# In[304]:


null_values(previous_application).sort_values(ascending=False)


# In[305]:


# Storing null values of columns in a variable named Null column
Null_column_previous = null_values(previous_application)


# In[306]:


Null_column_previous.sort_values(ascending=False)


# In[307]:


len(Null_column_previous)


# **There are 37 columns having null values in the previous_application dataset**

# In[308]:


#Plotting percentage of null values in columns in previous_application datset
plt.figure(figsize = (10,3), dpi=100)
Null_column_previous.plot(kind = "bar")
plt.title("Null values in columns")
plt.xlabel('Percentage of Null values')
plt.show()


# **Graph of null values in colunms in previous_application dataset**

# In[309]:


#creating a variable Null_column_40 for storing null columns having missing values more than 40%

Null_column_40_previous = null_values(previous_application)[null_values(previous_application)>40]


# In[310]:


Null_column_40_previous


# In[311]:


len(Null_column_40_previous)


# In[312]:


print("Number of columns having missing values more than 40% :",len(Null_column_40_previous))


# In[313]:


# Now lets drop all the columns having missing values more than 40% that is 11 columns

previous_application.drop(columns = Null_column_40_previous.index, inplace = True)


# In[314]:


previous_application.shape


# # IMPUTING VALUES FOR previous_application dataset

# In[315]:


#Storing columns with more than 1 percent null values in Null_column_1_previous
Null_column_1_previous = null_values(previous_application)[null_values(previous_application)>1]


# In[316]:


Null_column_1_previous


# In[317]:



previous_application[["AMT_ANNUITY","AMT_GOODS_PRICE","CNT_PAYMENT"]].describe()


# In[318]:


previous_application.fillna(previous_application[["AMT_ANNUITY","AMT_GOODS_PRICE","CNT_PAYMENT"]].median(),inplace = True)


# In[319]:


null_values(previous_application).head(10)


# **After replacing null values with median we have achieved negligible null values in previous application dataset**

# In[320]:


previous_application


# In[321]:


#Age/Days columns are in -ve which needs to be converted to +ve value
previous_application.DAYS_DECISION = previous_application.DAYS_DECISION.abs()


# In[322]:


previous_application.DAYS_DECISION


# **All the negatives have been converted to positive values**

# # IDENTIFYING OUTLIERS

# In[323]:


#storing important variables in outlier_previous for identifying outliers
outlier_previous = ["AMT_GOODS_PRICE","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT"]


# In[324]:


#identifying outliers
plt.figure(figsize=[15,25])
for i,j in itertools.zip_longest(outlier_previous, range(len(outlier_previous))):
    plt.subplot(4,2,j+1)
    sns.boxplot(y = previous_application[i], orient = "h", color = "red")
    #plt.yticks(fontsize=6)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# **AMT_GOODS_PRICE,AMT_ANNUITY,AMT_APPLICATION,AMT_CREDIT all columns have outliers present**

# # MERGING DATA FRAMES AND ANALYSIS

# In[325]:


# merge both the dataframe on SK_ID_CURR with Inner Joins
loan_merge = pd.merge(application_data, previous_application, how='inner', on='SK_ID_CURR')
loan_merge.head()


# In[326]:


loan_merge.shape


# In[327]:


loan_merge.info('all')


# In[328]:


loan_merge.describe()


# In[329]:


#Bisecting the loan colums of merged dataframe for defaulters and non defaulters
Loan_0= loan_merge[loan_merge['TARGET']==0] # Non Defaulters
Loan_1 = loan_merge[loan_merge['TARGET']==1] # Defaulters


# In[330]:


def loan_merged(col,df,hue,palette,ylog,figsize):
    plt.figure(figsize=figsize)
    ax=sns.countplot(x=col, data=df,hue= hue,palette= palette,order=df[col].value_counts().index)
    

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)",fontsize=18)     
    else:
        plt.ylabel("Count",fontsize=18)       

    plt.title(col , fontsize=15) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


# In[331]:


#plotting graph between NAME_CASH_LOAN_PURPOSE and NAME_CONTRACT_STATUS of merged dataframe with respect to Non defaulters
loan_merged("NAME_CASH_LOAN_PURPOSE",Loan_0,"NAME_CONTRACT_STATUS",["#295939","#e40017","#64dfdf","#fff600"],True,(16,10))


# In[332]:


loan_merged("NAME_CASH_LOAN_PURPOSE",Loan_1,"NAME_CONTRACT_STATUS",["#295939","#e40017","#64dfdf","#fff600"],True,(16,10))


# **Analysis and insights from merged data**
# 
# **Loan purpose has high number of unknown values (XAP, XNA)**
# 
# **Loan taken for the purpose of Repairs looks to have highest default rate**
# 
# **Huge number application have been rejected by bank or refused by client which are applied for Repair or Other. from this we can infer that repair is considered high risk by bank. Also, either they are rejected or bank offers loan on high interest rate which is not feasible by the clients and they refuse the loan.**

# # CONCLUSION

# Non Defaulters Percentage is 91.93% and defaulters is 8.07%.
# 
# Females are less likely to default to loans.
# 
# People living in office apartments have lowest default rate.
# 
# People with Academic degree are least likely to default.
# 
# Most of the loans are taken by Laborers, followed by Sales staff.
# 
# Applicants with rating 1 are less likely to default.
# 
# The customers who owns real estate are double in numbers from those who don’t known and it owning a real estate does not affect the defaulting.
# 
# Applicants living in region rating 1 are less likely to default.
# 
# Most of the loan defaults are for customers whose income ranges between 1100000-1500000.
# 
# Most of the loans are taken by Laborers, followed by Sales staff.
# 
# Self employed people have relative high defaulting rate, to be safer side loan disbursement should be avoided or provide loan with higher interest rate to mitigate the risk of defaulting.
# 
# Most of the people application for loan are from Business Entity Type 3.
# 
# Loan taken for the purpose of Repairs looks to have highest default rate.
# 
# 
# Huge number application have been rejected by bank or refused by client which are applied for Repair or Other. from this we can infer that repair is considered high risk by bank. Also, either they are rejected or bank offers loan on high interest rate which is not feasible by the clients and they refuse the loan.
# 
# From the EDA CASE STUDY Analysis we have drawn useful insights about Non Defaulters and defaulters which will be useful lending of loans.
# 
# We can say that any applicant able to repay the loan should be lended the loan and the applicants not able to repay must be avoided the loan . Both, the criteria's are important and should be taken care of for better functioning and revenue of bank.
# 
