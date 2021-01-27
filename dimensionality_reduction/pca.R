# PCA

# Importing the dataset
dataset = read.csv('Wine.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# prcomp() function and principal() function, giving the rotate='none' can both also be used
# the difference in loadings for the functions is only in scaling, 
# which means that in prcomp(), the summation of loadings is set to 1, while principal() is not. 
# However, If the "rotate" parameter of principal() function is set to TRUE, then the loadings cannot be explained compared to prcomp()'s.
# In the prcomp() function, the summation of loadings equals 1; 
# besides, the PC1 column's vector times the PC2 column's vector equals 0, 
# which is the definition that PC1 should be perpendicular to PC2. 
# In this way, I can actually interpret the matrix vertically. 
# Also, I can explain the matrix horizontally that q1 is the linear combination of RC1, RC2. 
# Proving that I can interpret from both directions before rotation
# However, after the rotation we cannot interpret the matrix vertically anymore.

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
# this is to re-arrange the table, c() is to get the data in a vector type
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))