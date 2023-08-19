""" Task: 1: Movie Recommendation """

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # %matplotlib qt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

sns.set_style('darkgrid')
sns.set(color_codes=True) 
sns.set_context('paper')
sns.set_palette("rocket_r")

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Set max rows displayed in output to 25
pd.set_option("display.max_rows", 25)

def precision_recall(y_true, y_pred):
  """ Calculates the precision and recall for the minority class (1) """
  
  # Counters
  TP = 0; FP = 0; FN = 0
  
  # Calculate the metrics
  for (true, pred) in zip(ytest, y_pred):
    if true == pred == 1:
      TP += 1
    elif true == 0 and pred == 1:
      FP += 1
    elif true == 1 and pred == 0:
      FN += 1
    
  # Error handling for division by 0
  try:
      precision = TP / (TP + FP)
  except:
      precision = 1
  try:
      recall = TP / (TP + FN)
  except:
      recall = 1
      
  return precision, recall

def aoc(precision, recall):
  """ Calculates the area under the curve """
  res = 0
  for i in range(len(recall)):
    # There is not R_n-1 in this case
    if i == 0:
      res += recall[i] * precision[i]
    else:
      res += (recall[i] - recall[i-1]) * precision[i]
  return res

def thresholding(decisions, thresholds, ytest):
  """ Calculates the recall- and precision values while ajusting the threshold 
  for the hyperplane """
  
  precision_scores = []
  recall_scores = []

  # Metrics for each threshold
  for tau in thresholds:
      y_test_preds = []
      for dec in decisions:
          if dec > tau: # classified as member of class 1
              y_test_preds.append(1)
          else:
              y_test_preds.append(0)
              
      precision, recall = precision_recall(ytest, y_test_preds)

      recall_scores.append(recall)
      precision_scores.append(precision)
  
  return recall_scores, precision_scores

""" ---------------- Loading the data sets ----------------- """

try: # results were previously stored
  test_set = pd.read_csv('./ml-1m/test_set.csv')
  training_set = pd.read_csv('./ml-1m/training_set.csv')
  
except: # create data anew
  df_ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, 
                          names=['UserID', 'MovieID', 'Rating'], 
                          usecols=[0,1,2])

  df_movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, 
                          names=['MovieID', 'Title', 'Genres'])

  df_users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, 
                        names=['UserID', 'Gender', 'Age', 'Occupation'], 
                        usecols=[0,1,2,3])
                        
  """ ------------- Data Preparation ------------------------ """
  
  def convert_rating(x):
      """ converts the movie rating to binary labels.
      1 --> user liked the movie, 0 --> user didn't like it """
      return 1 if x == 5 or x == 4 else 0

  def convert_gender(str):
      """ converts the gender to binary labels.
      0 --> M, 1 --> F """
      return 0 if str == 'M' else 1

  # Contains all genres
  dic = {'Action':0, 'Adventure':1, 'Animation':2, "Children's":3, 'Comedy':4,
        'Crime':5, 'Documentary':6, 'Drama':7, 'Fantasy':8, 'Film-Noir':9,
        'Horror':10, 'Musical':11, 'Mystery':12, 'Romance':13, 'Sci-Fi':14,
        'Thriller':15, 'War':16, 'Western':17}

  # Takes first (primary) genre and converts it to numeric value
  convert_genre = lambda x: dic[x.split('|')[0]]

  # Convertions of features to numerical values
  df_ratings['Rating'] = df_ratings['Rating'].apply(convert_rating)
  df_users['Gender'] = df_users['Gender'].apply(convert_gender)
  df_movies['Genres'] = df_movies['Genres'].apply(convert_genre)

  # Adding the number ratings for each user to the data frame
  df_users['Ratings'] = df_ratings.groupby('UserID')['MovieID'].count().reset_index(name="count")['count']

  # Removing all users with less than 200 ratings
  df_users = df_users[df_users['Ratings'] >= 200]

  """ ------------- Feature Vector Creation ----------------- """
  
  # Inner join of ratings and movies on MovieID to get rating for every movie
  new = pd.merge(df_ratings,df_movies,on='MovieID')

  # Only selecting ratings of users with at least 200 ratings
  new = new[new['UserID'].isin(df_users['UserID'])]
  new = new.drop('Title', axis=1)

  # Inner join to get user information (Gender, Age, Occupation)
  new = pd.merge(new, df_users, on='UserID').drop('Ratings', axis=1)

  # Separating data into test- and training set via UserID
  test_set = new[new['UserID'] <= 1000]
  training_set = new[new['UserID'] > 1000]
  
  test_set.to_csv('./ml-1m/test_set.csv', index=False)
  training_set.to_csv('./ml-1m/training_set.csv', index=False)

# Data splitting
xtest = test_set[['MovieID', 'Genres', 'Gender', 'Age']]
ytest = test_set['Rating']
xtrain = training_set[['MovieID', 'Genres', 'Gender', 'Age']]
ytrain = training_set['Rating']

# Normalization
scaler = MinMaxScaler().fit(xtrain) 
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

print('Size of training set: ', len(xtrain),
      '\nSize of test set: ', len(xtest),
      '\nNumber of observed features: ', xtrain.shape[1], "\n")

#%%
""" ------------------ Selecting hyperparameters --------------------- """

# SVM parameter

param_grid = [
  {'C': np.arange(0.1,0.5,0.1)}
 ]

# Evaluating model with different parameter combinations
svm = GridSearchCV(LinearSVC(), param_grid, n_jobs=-1).fit(xtrain, ytrain)
print("Selecting best hyperparameter:")
print("Best SVM-Classifier: ", svm.best_estimator_, "\n")

# MLP parameter
param_grid = [
  {'hidden_layer_sizes': [(20,20),
                           (25,25)]
   }
  ]
"""
                          (10,50), (10, 100) 
                          (20,50),(20,100),
                          (25,50), (25,100)]}
 ]
"""
#%%
# Uses k randomly chosen samples for fitting the model
k = int(ytrain.shape[0] / 5) # fraction of the training-set
ran = np.random.choice(range(0, ytrain.shape[0]), k, replace=False)
clf = MLPClassifier()
ytrain = ytrain.reset_index(drop=True).to_numpy()

# Evaluates model with different parameter combinations
mlp_grid = GridSearchCV(clf, param_grid, n_jobs=-1).fit(xtrain[ran], ytrain[ran])
print("Selecting best hyperparameter:")
print("Best MLP-Classifier: ", mlp_grid.best_estimator_, "\n")
#%%

# Training optimal classifiers, using random_state to fix weight and 
# bias initialization so to have reproducable results
# (25,25) hard-coded because GridSearchCV only uses random subset to train MLP
mlp = MLPClassifier((25,25), random_state=2)
mlp.fit(xtrain, ytrain)

# Predicted values
y_pred_svm =  svm.best_estimator_.predict(xtest)
y_pred_mlp =  mlp.predict(xtest)

correct = np.sum(ytest == y_pred_mlp) / ytest.shape[0]

print(np.round(correct * 100, 1), "% test samples correctly classified using MLP Classifier")

correct = np.sum(ytest == y_pred_svm) / ytest.shape[0]

print(np.round(correct * 100, 1), "% test samples correctly classified using SVM Classifier")

#%%
""" ---------------- Precision Recall - Curves and APs -------------- """

# Confidence values for both classifiers
dec_pred_svm = svm.best_estimator_.decision_function(xtest)
dec_pred_mlp = mlp.predict_proba(xtest)[:,1] # class 1

# Define probability thresholds to use
thresholds_svm = np.linspace(max(dec_pred_svm), min(dec_pred_svm), 50)
thresholds_mlp = np.linspace(max(dec_pred_mlp), min(dec_pred_mlp), 50)

# Metrics for both classifiers
recall_scores_svm, precision_scores_svm = thresholding(dec_pred_svm, 
                                                       thresholds_svm, 
                                                       y_pred_svm)

recall_scores_mlp, precision_scores_mlp = thresholding(dec_pred_mlp,
                                                       thresholds_mlp,
                                                       y_pred_mlp)

# plot of the precision recall curves
plt.figure()
plt.title('Precision Recall Curve', fontsize=12 ,
            fontweight='bold')
plt.plot(recall_scores_mlp, precision_scores_mlp, lw=2, c='r', 
         label='MLPClassifier: AP= '+str(round(aoc(precision_scores_mlp, recall_scores_mlp),
                                               2)), alpha=0.7)
plt.plot(recall_scores_svm, precision_scores_svm, lw=2, c='#e17701', ls='dotted', 
         label='LinearSVM: AP='+str(round(aoc(precision_scores_svm, recall_scores_svm), 2))
         , alpha=0.7)

plt.xlabel("Recall", fontweight='bold')
plt.ylabel("Precision" , fontweight='bold')
plt.ylim(0,1)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('1.3.png', dpi=600)

plt.show()