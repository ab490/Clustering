#Name: Anooshka Bajaj

import pandas as pd
train_data=pd.read_csv(r'E:\mnist-tsne-train.csv')
test_data=pd.read_csv(r'E:\mnist-tsne-test.csv')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy import spatial as spatial

train = train_data.iloc[:, :-1] 
test= test_data.iloc[:, :-1]


#1
def KNN(K):
    print('\nK:',K)
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(train)
    kmeans_prediction = kmeans.predict(train)
    cluster_centre_knn = pd.DataFrame(kmeans.cluster_centers_)
    
    #1(a)
    plt.scatter(train["dimention 1"], train["dimension 2"], c=kmeans_prediction)
    plt.scatter(cluster_centre_knn[0], cluster_centre_knn[1], color="red")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("KNN Clustering Training Data")
    plt.show()
    
    #1(b)
    def purity_score(y_true, y_pred):
        contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)            #compute contingency matrix (also called confusion matrix)
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)                    #find optimal one-to-one mapping between cluster labels and true labels
        return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)      #return cluster accuracy
    
    print('Purity Score:',purity_score(train_data["labels"],kmeans_prediction))
    
    #1(c)
    kmeans_test_prediction = kmeans.predict(test)
    plt.scatter(test["dimention 1"], test["dimention 2"], c=kmeans_test_prediction)
    plt.scatter(cluster_centre_knn[0], cluster_centre_knn[1], color="red")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("KNN Clustering Test Data")
    plt.show()
    
    #1(d)
    print('Purity Score:',purity_score(test_data["labels"],kmeans_test_prediction))

KNN(10)


#2
def GMM(K):
    print('\nK:',K)
    gmm = GaussianMixture(n_components = K)
    gmm.fit(train)
    GMM_prediction = gmm.predict(train)
    cluster_centre_gmm = pd.DataFrame(gmm.means_)
    
    #2(a)
    plt.scatter(train["dimention 1"], train["dimension 2"], c=GMM_prediction)
    plt.scatter(cluster_centre_gmm[0], cluster_centre_gmm[1], color="red")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("GMM Clustering Training Data")
    plt.show()
    
    #2(b)
    def purity_score(y_true, y_pred):
        contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)            #compute contingency matrix (also called confusion matrix)
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)                    #find optimal one-to-one mapping between cluster labels and true labels
        return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)  
    print("Purity Score:", purity_score(train_data["labels"], GMM_prediction))
    
    #2(c)
    GMM_test_prediction = gmm.predict(test)
    plt.scatter(test["dimention 1"], test["dimention 2"], c=GMM_test_prediction)
    plt.scatter(cluster_centre_gmm[0], cluster_centre_gmm[1], color="red")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("GMM Clustering Test Data")
    plt.show()
    
    #2(d)
    print("Purity Score:", purity_score(test_data["labels"], GMM_test_prediction))

GMM(10)


#3
def dbscan(eps, minsamples):
    print('\neps:',eps,'\nmin_samples:',minsamples)
    dbscan_model=DBSCAN(eps=eps, min_samples=minsamples).fit(train)
    DBSCAN_predictions = dbscan_model.labels_

#3(a)
    plt.figure()
    plt.scatter(train["dimention 1"], train["dimension 2"], c=DBSCAN_predictions)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("DBSCAN Clustering Train Data")
    plt.show()

#3(b)
    def purity_score(y_true, y_pred):
        contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)            #compute contingency matrix (also called confusion matrix)
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)                    #find optimal one-to-one mapping between cluster labels and true labels
        return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix) 
    print("Purity Score:",purity_score(train_data["labels"],DBSCAN_predictions))

#3(c)
    def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
        y_new = np.ones(shape=len(X_new), dtype=int) * -1
        for j, x_new in enumerate(X_new):
            for i, x_core in enumerate(dbscan_model.components_):
                if metric(x_new, x_core) < dbscan_model.eps:
                    y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                    break
        return y_new

    DBSCAN_test_prediction = dbscan_predict(dbscan_model, np.array(test), metric=spatial.distance.euclidean)
    plt.figure()
    plt.scatter(test["dimention 1"], test["dimention 2"], c=DBSCAN_test_prediction)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(" DBSCAN Clustering Test Data ")
    plt.show()

#3(d)
    print("Purity Score:",purity_score(test_data["labels"],DBSCAN_test_prediction))

dbscan(5,10)


#A
#1
K=[2,5,8,12,18,20]  
for m in K:
    KNN(m)
    
#2
K=[2,5,8,12,18,20]  
for m in K:   
    GMM(m)

#optimum number of clusters using elbow method
def purity_score(y_true, y_pred):
        contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)            #compute contingency matrix (also called confusion matrix)
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)                    #find optimal one-to-one mapping between cluster labels and true labels
        return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

k=[2,5,8,12,18,20]  

#K-means 
def distortion(k):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(train)
    return kmeans.inertia_

dist=[]
for i in k:
    dist.append(distortion(i))
    
plt.plot(k,dist,marker='o')
plt.xlabel('k')
plt.ylabel('distortion measure')
plt.title('elbow method for K-means clustering')
plt.show()
print('Optimum number of clusters using elbow method for K-means clustering =',8)     #value of t for which the elbow bends is the optimum no. of clusters

#GMM    
def loglikelihood(k):
    gmm = GaussianMixture(n_components = k,random_state=42)
    gmm.fit(train)
    return gmm.lower_bound_

totallog=[]
for i in k:
    totallog.append(loglikelihood(i))
    
plt.plot(k,totallog,marker = 'o')
plt.xlabel('k')
plt.ylabel('log likelihood')
plt.title('elbow method for GMM')
plt.show()
print('Optimum number of clusters using elbow method for GMM clustering =',8) 
   
#B
epsilon=[1,5,10]
for i in epsilon:
    dbscan(i,10) 

minsamples=[1,10,30,50]
for j in minsamples:
    dbscan(5,j)
  





