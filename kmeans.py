import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(12345)



# Function for loading the iris data
# load_data returns a 2D numpy array where each row is an example
#  and each column is a given feature.
def load_data():
    iris = datasets.load_iris()
    return iris.data

# Assign labels to each example given the center of each cluster
def assign_labels(X, centers):
    labels =[]  # empty list that would store all the labels for the data points
    for i in X:     # each data point
        dist =[]    # defined to temporarily store the distances of the points from centroid
        for j in centers:   # taking each cenroid at a time
            sub = np.subtract(i,j)  # subtracting the datapoint from the centroid
            tot = 0
            for val in sub:     # taking each feature value at a time
                tot = tot + val**2
            euclidean = np.sqrt(tot)    # calculates the euclidean dist for each  point
            dist.append(euclidean)      # appends the distances
        label = "K"+str(np.argmin(dist)+1)  #Assigns a label, according to the minimum dist
        labels.append(label)        # appends all labels
    return labels


# Calculate the center of each cluster given the label of each example
def calculate_centers(X, labels): 
    centroids = []  # empty list defined to store the centroids
    Lab = list(sorted(set(labels)))  # getting the sorted unique labels from the labels list 
    joint = []      #empty list defined to store the point and respective label tuple
    #points_cluster =[]
    for x,l in zip(X,labels): # mapping each datapoint to its label
        joint.append((x,l))
    # clubs all datapoints of a perticular label in one cluster
    for label in Lab:
        cluster =[]
        for i in joint:
            if i[1]==label:
                cluster.append(i[0])
        #points_cluster.append(len(cluster))
        centroids.append(np.mean(cluster,axis = 0))
    centroids = [val.tolist() for val in centroids] # converts arrays to lists
    #print("points in each cluster = "+ str(points_cluster))
    return centroids

# Test if the algorithm has converged
# Should return a bool stating if the algorithm has converged or not.
def test_convergence(old_centers, new_centers):
    if old_centers == new_centers:
        return True
    else:
        return False

# Evaluate the preformance of the current clusters
# This function should return the total mean squared error of the given clusters
def evaluate_performance(X, labels, centers):
    jointX =[]      # defined to store the datapoint and label tupple
    unique_labels = list(sorted(set(labels))) # getting the sorted unique labels from the labels list 
   
    for x,l in zip(X,labels):  #pairs the datapoint and its respective label
        jointX.append((x,l))
    j = 0
    euclidean = 0
    for label in unique_labels:     #puts all datapoints having same label into one cluster
        cluster =[]
        for i in jointX:
            if i[1]==label:
                cluster.append(i[0])
        #for each cluster, calculates the total euclidean distance 
        #returns the total euclidean distances of all the clusters
        m = centers[j]
        j += 1
        for x in cluster:
                sub = np.subtract(x,m)
                tot = 0
                for val in sub:     # each feature
                    tot = tot + val**2
                euclidean = euclidean + tot
    return euclidean

# Algorithm for preforming K-means clustering on the give dataset
def k_means(X, K):
    # Initially selects K points from the data in random
    centers=[]
    for i in np.random.randint(1,149,K): #random initialization of k centroids
        centers.append(X[i])
    centers = [val.tolist() for val in centers]
    # getting labels for each data point in the data set
    labels = assign_labels(X, centers)
    # calculating new centroids
    new_center = calculate_centers(X, labels)
   #testing for convergence
    while (test_convergence(centers, new_center)!=True): #looping until it converges
        centers = new_center
        labels = assign_labels(X, centers)
        new_center = calculate_centers(X, labels)
    return(X,labels,evaluate_performance(X, labels, centers))

iris = load_data()
cost =[]    #empty list defined to store the SSE values for different values of K
k_values =[]    #empty list defined to store different values of K
for k in range(1,10):
    (x,L,E) = k_means(iris, k)
    cost.append(E)
    k_values.append(k)
#plotting the SSE against values of K
plt.plot(k_values,cost)
plt.xlabel("value of K")
plt.ylabel("SSE")
plt.title("graph of SSE over K")
plt.show()

