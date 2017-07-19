# homework-ii-mfigueroa94
name: Michael Figueroa
uni: mf3090 

Prediciting Rent Price:

1) I first began manually selecting features by looking at the feature descriptions. 
   I noticed the feature (sc116) encodes whether an apartment is owner occupied,
   occupied rent free, or paid in cash rent. I removed the rows which were owner 
   occupied or occupied rent free under the assumtion that the people who fill the survey
   and are paying cash rent will have features with the most relevant information for our prediction.
   I noticed that after removing these rows, my response variables rows with missing values were also
   deleted.

2) I removed features related to demographics, social security, information specific to householder
   and any other features which I deemed were specific to an apartment currently rented. Note: I was fairly
   generous with keeping features if I was unsure if they had good prediction power. Having many features may
   make the model very complex, but I will use model based selection of features later on , in addition to an
   estimator such as Lasso Regression which drives the alpha values of non-important features down to zero.


2) After manually removing features I was left with 86 features with missing data. 
	Here is a list of the features that I was left with:
   	['boro', 'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6', 'uf1_7',
       'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13', 'uf1_14',
       'uf1_15', 'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18', 'uf1_19', 'uf1_20',
       'uf1_21', 'uf1_22', 'sc23', 'sc24', 'sc36', 'sc37', 'sc38', 'uf48',
       'sc147', 'uf11', 'sc149', 'sc173', 'sc171', 'sc150', 'sc151', 'sc152',
       'sc153', 'sc154', 'sc155', 'sc156', 'sc157', 'sc158', 'sc159', 'sc161',
       'sc164', 'sc166', 'uf17', 'sc185', 'sc186', 'sc197', 'sc198', 'sc187',
       'sc188', 'sc571', 'sc189', 'sc190', 'sc191', 'sc192', 'sc193', 'sc194',
       '#sc196', 'uf19', 'new_csr', 'rec15', 'sc26', 'uf23', 'rec21', 'rec62',
       'rec64', 'rec54', 'rec53', 'cd', 'hflag6', 'hflag3', 'hflag14',
       'hflag16', 'hflag7', 'hflag9', 'hflag10', 'hflag91', 'hflag11',
       'hflag12', 'hflag4', 'hflag18']

3) The next step was to think about how I was going to impute my dataset for missing values 
	such as 9 or 8 in my data set. I noticed the features I was left with were all categorical 
	with the exception of my response variable 'uf17'. For my categorical features in my data set, 
	I noticed that when I have missing data, It is already encoded using a seperate category. I chose to 
  leave my categorical features this way in order to capture relevant information tied to a missing value. After 
  I One-Hot-Encode my categorical features, it will create a new feature that identifies 
	if data was missing. (Often missingness is informative). NOTE: I also attempted using imputation (at end of the file)


4) Next I used One-Hot-Encoding to handle my categorical variables. Our categories are numeric so I specify get_dummies to convert all
  columns except my response column to Strings. This will allow my integers to be treated as categories.
   on all columns except my response variable to be applied.

5) I used a pipeline which ensures that my transformations such as MinMaxScaler are performed within the cross validation
	loop. This prevents data leakage by fitting my transformation on the training folds and testing on validation.

6) I chose LassoCV as my estimator to train on. This does automatic feature selection by driving certain alphas down to 0.
   It also finds the best alphas using cross validation which is equivalent to doing gridSearchCV() to find parameters. I performed 
   10-fold cross-validation on my alpha parameters to get a mean cross validation accuracy of 0.503812165607 on the training set.

7) My R^2 for my model is 0.5179501731029843. I validated my model through several comparisons with other models. I compared my MinMaxScaler to StandardScaler and got a slightly higher accuracy on the test set with MinMaxScaler. I also tried other models such as Ridge, which gave me a
   slightly lower predictive accuracy on the training and the test set. 
   My summary statistics for ElasticNetCV was train: 0.402421398062 test: 0.417225387247 which was worse than both Lasso and Ridge.

NOTE: I double checked if most frequent imputation on categorical features would improve my predictive accuracy, given my preprocessing techniques. However, 
after using most_frequent imputation I got a training accuracy of train: 0.308191327168 and testing accuracy test: 0.31045528015.  This is how I validated that my first method without imputation was more effective. The following code shows 
the Imputation that I did:

# Look through current features and replace missing values with np.nan
# NOTE: FOR UF1_1 TO UF1_22 I ONLY REMOVED "RESPONSE NOT RECORDED" AND KEPT 
# "CONDITION NOT REPORTED" TO AVOID ONLY 1 CATEGORY AND LOSS OF INFORMATION
rent_data.ix[:, 'uf1_1':'uf1_22'] = rent_data.ix[:,'uf1_1':'uf1_22'].replace([9], np.nan)
rent_data['sc23'] = rent_data['sc23'].replace([8], np.nan)
rent_data['sc24'] = rent_data['sc24'].replace([8], np.nan)
rent_data['sc36'] = rent_data['sc36'].replace([8], np.nan)
rent_data['sc37'] = rent_data['sc37'].replace([8], np.nan)
rent_data['sc38'] = rent_data['sc38'].replace([8], np.nan)
rent_data['sc173'] = rent_data['sc173'].replace([8,3], np.nan)
rent_data['sc171'] = rent_data['sc171'].replace([8,3], np.nan)
rent_data['sc154'] = rent_data['sc154'].replace([8,9], np.nan)
rent_data['sc156'] = rent_data['sc156'].replace([9], np.nan)
rent_data['sc157'] = rent_data['sc157'].replace([8,9], np.nan)
rent_data['sc185'] = rent_data['sc185'].replace([8], np.nan)
rent_data['sc186'] = rent_data['sc186'].replace([8], np.nan)
rent_data['sc197'] = rent_data['sc197'].replace([4,8], np.nan)
rent_data['sc198'] = rent_data['sc198'].replace([8], np.nan)
rent_data['sc187'] = rent_data['sc187'].replace([8], np.nan)
rent_data['sc196'] = rent_data['sc196'].replace([8], np.nan)
rent_data['rec15'] = rent_data['rec15'].replace([10,11,12], np.nan)
rent_data['rec21'] = rent_data['rec21'].replace([8], np.nan)
rent_data['rec54'] = rent_data['rec54'].replace([7], np.nan)
rent_data['rec53'] = rent_data['sc196'].replace([9], np.nan)
# Replace all 2's in hflag to 1 (since 2 is the default value of 1)
rent_data.ix[:,'hflag6':] = rent_data.ix[:,'hflag6':].replace([2], [1])

rent_data = rent_data.drop('uf17', 1)
# Impute missing data using the most frequent
rent_data = pd.DataFrame(Imputer(missing_values = np.nan, strategy = "most_frequent", axis = 0).fit_transform(rent_data))

# One-Hot-Encoder
rent_dummies = pd.get_dummies(rent_data)
