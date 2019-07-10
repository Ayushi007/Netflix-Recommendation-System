# from google.colab import drive
# drive.mount('/content/drive')

# %matplotlib inline
import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matrix=pd.read_csv('drive/Team Drives/ECS 271/R.csv')
# print(matrix.shape)
# temp= pd.read_csv('drive/Team Drives/ECS 271/test.csv', names=['movie-id', 'user-id', 'rating', 'date'])
# #temp.columns = ['movie-id', 'user-id', 'rating']
# temp.head(10)

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            #if (i+1) % 10 == 0:
            print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process



    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])
            #print(self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)

data = pd.read_csv('path/train.csv')
data.columns= ["movieId","customer-id","rating","date"]
result=pd.DataFrame(data)
#movies_train= np.unique(result.movieId.values)

movie_col=result.columns.values
user_row=result.index.values

train=result[200001:][:]
test =result[:200000][:]
#train=result[:][:]
teste =test.copy()


#print(teste)
test1=test.pivot_table(index='customer-id', columns='movieId', values='rating').fillna(0)

T=np.array(test1)
# print(T.shape)
test['rating'] = 0
frames=[train, test]
train1=pd.concat(frames)

print(train1.shape)
#print(train1)
final=train1.pivot_table(index='customer-id', columns='movieId', values='rating').fillna(0)


final1=final.copy()
R = np.array(final)
#print(R)
print(R.shape)

mf = MF(R, K=4, alpha=0.006, beta=0.001, iterations=60)
training_process = mf.train()
#print()
#print("P x Q:")
#print(mf.full_matrix())
R=mf.full_matrix()

count = 0
size=0
for i in range(len(R)):
    for j in range(len(R[i])):
        size+=1
        if R[i][j]>5 :
            #print(R[i][j])
            count +=1
            R[i][j] = 5
        if R[i][j]<0:
            R[i][j]=0
print(count, size)

print(R)
final2=pd.DataFrame(R, index=final1.index, columns=final1.columns)
print(final2.shape)

cols=test1.columns.values

rows=test1.index.values
#print(rows)

error =0
entries=0
i=0
#print(teste)
for index, row in teste.iterrows():
  #print(row['movie-id'], row['customer-id'])
  movie=row['movieId']
  cust=row['customer-id']
  rat=row['rating']
  error+=pow(rat-final2.loc[cust, movie], 2)
  #print(rat,final2.loc[cust, movie],error)
  entries+=1
error/=entries
print(np.sqrt(error))
######
""" Error calculated with only matrix decomposition of the train data"""


###########################################
######################   Method 2  ########################
###########################################

final3=final2.copy()
movies=final3.columns.values
users=final3.index.values

frame_kmeans=pd.read_csv('path/metadata_new2.csv')
original_title = pd.read_csv('path/title_original.csv')

title=pd.merge(frame_kmeans, original_title, how='right', on=['title'])


unique_cluster=sorted(title.cluster.unique())
from __future__ import print_function

print("Top terms per cluster:")
movie_cluster={}

count=0

for item in unique_cluster:
  cluster_titles=title.loc[(title['cluster']==item) & (title['movie_id'].isin(movies))]['movie_id'].values.tolist()
  if len(cluster_titles)!=0:
    print(cluster_titles)
    key=cluster_titles[0]
    value=set(cluster_titles[:])
    movie_cluster.update({key: value})
  #print(key, value)
  print()
  count+=1

count=0
for key, value in movie_cluster.items():
  count+=1
  for user in users:
    sum=0
    for val in value:
      sum+=final3.loc[user][val]
    avg=sum/len(value)

    for val in value:
      final3.loc[user][val]=avg
  print(count)

error1=0
entries1=0
j=0
for index, row in teste.iterrows():
  #print(row['movie-id'], row['customer-id'])
  movie=row['movie-id']
  cust=row['customer-id']
  rat=row['rating']
  error1+=pow(rat-final3.loc[cust, movie], 2)
  entries1+=1
error1/=entries1
print(np.sqrt(error1))
################
"""Error calculated with matrix decomposition and clusters on basis of movie data"""
