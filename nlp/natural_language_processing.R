# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote='')
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
# to turn the dataset into a corpus object type
corpus = VCorpus(VectorSource(dataset_original$Review))
# to turn all the capitial words to a lowercase word
# tm_map is a mapping function to map the data to the function at the second argument
corpus = tm_map(corpus, content_transformer(tolower))
# to remove all the numbers
corpus = tm_map(corpus, removeNumbers)
# to remove all the punctuations
corpus = tm_map(corpus, removePunctuation)
# to remove all the english stop words, the stop words are from the Snowball library
corpus = tm_map(corpus, removeWords, stopwords())
# to use a stemmer to find the root of the word
corpus = tm_map(corpus, stemDocument)
# to strip off extra white space
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
# to decrease the sparsity, keep the most frequent words
dtm = removeSparseTerms(dtm, 0.999)
# create a data frame for the machine learning model usage
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)