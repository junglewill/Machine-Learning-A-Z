# Machine-Learning-A-Z
Exercises from taking Udemy online course - Machine Learning A-Z: Hands-on Python & R in Data Science excluding the deep learning part. The notes cover what I learnt from the online courses about machine learning models' deployment on Python.

The repository includes a Python_notes.ipynb to list the important take-aways from this course, and all the R data modeling and visualization templates that can be directly used:
### Regression in R
* Simple Linear Regression (train_test_split using sample.split function from <b>caTools</b> library, linear regression using lm function)
* Polynomial Linear Regression (polynomial linear regression using lm function)
* Multi-Linear Regression (multi-linear regression using lm function)
* Support Vector Regression (svr using svm function with type = 'eps-regression' from <b>e1071</b> library)
* Decision Tree Regression (decision tree using rpart function from <b>rpart</b> library) 
* Random Forest Regression (random forest using randomForest function with ntree = (num of trees) from <b>randomForest</b> library)

### Classification in R
* Logistic Regression (confusion matrix using table function, logistic regression using glm function)
* K-Nearest Neighbors (knn using knn function from <b>class</b> library)
* Support Vector Machine (svm using svm function with type = 'C-classification', kernal = 'linear' from <b>e1071</b> library)
* kernel SVM (svm using svm function with type = 'C-classification', kernal = 'radial' from <b>e1071</b> library)
* Naive Bayes (naive bayes using naiveBayes function from <b>e1071</b> library) remember: the y for naiveBayes must be factor encoded to be recognized by the function
* Decision Tree Classifier (decision tree using rpart function with type = 'class' from <b>rpart</b> library) to print the tree out, use <b>plot(classifier)</b> and <b>text(classifier)</b> to see visualization
* Random Forest Classifier (random forest using randomForest function with ntree = (num of trees) from <b>randomForest</b> library)

### Clustering in R
* K means (kmeans function with X, 5, iter.max=300, nstart=10) (to visualize the clustering, use clusplot function from <b>cluster</b> library)
* Hierarchical Clustering (dendrogram using hclust(dist(X, method='euclidean'), method='ward.D')) (use cuttree(hc, 5) to get the prediction) (to visualize the clustering, use clusplot function from <b>cluster</b> library)

### Association Rule Learning in Python & R
in R, use read.transactions instead of read.csv to create the sparse matrix
* Apriori (apriori function with parameter = list(support = 0.004, confidence = 0.2) from <b>arules</b> library)

* Principle Component Analysis(PCA) in Python and R
* Tensorflow in Python and R
