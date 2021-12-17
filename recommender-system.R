##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################


#
# FUNCTIONS
# FOR RUNNING
# ALGORITHM
#

# Sets arranged numbers for users and movies from 1 to entity number
setArrangedNumberToUsersAndMovies <- function(dataset) {
  grouped_u_ids <- dataset %>% group_by(userId) %>% summarise(user_n=n())
  grouped_u_ids$user_number <- seq.int(nrow(grouped_u_ids))
  
  grouped_m_ids <- dataset %>% group_by(movieId) %>% summarise(movie_n=n())
  grouped_m_ids$movie_number <- seq.int(nrow(grouped_m_ids))
  
  dataset %>%
    left_join(grouped_u_ids, by="userId") %>%
    left_join(grouped_m_ids, by="movieId") %>%
    select(-user_n,-movie_n)
}

# Applies arranged number from one dataset to another
applyUsersAndMoviesNumberFromOneDatasetToAnother <- function(datasetWithNumbers, datasetWithoutNumbers) {
  train_user_ids_and_numbers <- datasetWithNumbers %>% select(userId, user_number)
  train_movie_ids_and_numbers <- datasetWithNumbers %>% select(movieId, movie_number)
  
  datasetWithoutNumbers %>%
    left_join(distinct(train_user_ids_and_numbers), by="userId") %>%
    left_join(distinct(train_movie_ids_and_numbers), by="movieId")
}


# Randomly initializes 1 vector for factorization matrix
# Depends on matrix rank
randomlyInitialiseFactorizationVector <- function(length, matrixRank = 2, maxRating = 5) {
  avgFactor <- maxRating ** (1/(matrixRank*2))
  
  sample(c(avgFactor - 0.1, avgFactor, avgFactor + 0.1), size = length, replace = T, prob = c(0.33,0.33,0.33))
}

# Initializes factorization matrix for users by given rows and columns number
initialiseUserFactorizationMatrix <- function(nrow, ncol = 2) {
  userFactorizationMatrix <- matrix(nrow = nrow, ncol = ncol)
  
  userFactorizationMatrix[,1] <- vector(mode = 'numeric', length = nrow)
  
  if (ncol > 1) {
    for (col in 2:ncol) {
      userFactorizationMatrix[,col] <- randomlyInitialiseFactorizationVector(length = nrow, matrixRank = ncol)
    }
  }
  
  userFactorizationMatrix
}


# Initializes factorization matrix for movies by given rows and columns number
initialiseMovieFactorizationMatrix <- function(nrow, ncol = 2) {
  movieFactorizationMatrix <- matrix(nrow = nrow, ncol = ncol)
  
  for (col in 1:ncol) {
    movieFactorizationMatrix[,col] <- randomlyInitialiseFactorizationVector(length = nrow, matrixRank = ncol)
  }
  
  movieFactorizationMatrix
}

# Makes one epoch optimization for user factorization matrix
update2ColUserFactorizationMatrix <- function(dataset, lambda, userMatrix, movieMatrix) {
  userMatrix[,1] <- dataset %>% 
    mutate(
      numerator = movieMatrix[movie_number,1] * (rating - movieMatrix[movie_number,2] * userMatrix[user_number,2]),
      denominator = movieMatrix[movie_number,1]**2
    ) %>%
    group_by(user_number) %>%
    summarise(new_u_1 = sum(numerator)/(sum(denominator)+ lambda)) %>%
    arrange(user_number) %>%
    .$new_u_1
  
  userMatrix[,2] <- dataset %>% 
    mutate(
      numerator = movieMatrix[movie_number,2] * (rating - movieMatrix[movie_number,1] * userMatrix[user_number,1]),
      denominator = movieMatrix[movie_number,2]**2
    ) %>%
    group_by(user_number) %>%
    summarise(new_u_2 = sum(numerator)/(sum(denominator)+ lambda)) %>%
    arrange(user_number) %>%
    .$new_u_2
  
  userMatrix
}

# Makes one epoch optimization for movie factorization matrix
update2ColMovieFactorizationMatrix <- function(dataset, lambda, userMatrix, movieMatrix) {
  movieMatrix[,1] <- dataset %>% 
    mutate(
      numerator = userMatrix[user_number,1] * (rating - userMatrix[user_number,2] * movieMatrix[movie_number,2]),
      denominator = userMatrix[user_number,1]**2
    ) %>%
    group_by(movie_number) %>%
    summarise(new_m_1 = sum(numerator)/(sum(denominator)+ lambda)) %>%
    arrange(movie_number) %>%
    .$new_m_1
  
  movieMatrix[,2] <- dataset %>% 
    mutate(
      numerator = userMatrix[user_number,2] * (rating - userMatrix[user_number,1] * movieMatrix[movie_number,1]),
      denominator = userMatrix[user_number,2]**2
    ) %>%
    group_by(movie_number) %>%
    summarise(new_m_2 = sum(numerator)/(sum(denominator)+ lambda)) %>%
    arrange(movie_number) %>%
    .$new_m_2
  
  movieMatrix
}

# Returns dataset with estimations based on provided factorization matrices
get2RankEstimations <- function(dataset, userMatrix, movieMatrix) {
  datasetWithEstimations <- dataset %>%
    mutate(
      u_1 = ifelse(is.na(user_number), mean(userMatrix[,1]), userMatrix[user_number,1]),
      u_2 = ifelse(is.na(user_number), mean(userMatrix[,2]), userMatrix[user_number,2]),
      m_1 = ifelse(is.na(movie_number), mean(movieMatrix[,1]), movieMatrix[movie_number,1]),
      m_2 = ifelse(is.na(movie_number), mean(movieMatrix[,2]), movieMatrix[movie_number,2]),
    ) %>%
    mutate(rating_hat = u_1 * m_1 + u_2 * m_2) %>%
    mutate(rating_hat = ifelse(rating_hat > 5, 5, rating_hat))
  
  datasetWithEstimations
}

# Function to check the results of ML
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
###################################################


# 1. TRAINING DATA PARTITION
test_train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)

edx_train <- edx[-test_train_index,]
edx_validation <- edx[test_train_index,]


#
# 2. DATA WRANGLING
# Idea is to set arranged numbers to users and movies
#
#
edx_train_with_numbers <- setArrangedNumberToUsersAndMovies(edx_train)

edx_validation_with_numbers <- applyUsersAndMoviesNumberFromOneDatasetToAnother(edx_train_with_numbers, edx_validation)


#
# 3. RUN REGULARIZATION TRAINING
#
USERS_REG <- edx_train_with_numbers %>% group_by(userId) %>% summarise(n=n()) %>% nrow()
MOVIES_REG <- edx_train_with_numbers %>% group_by(movieId) %>% summarise(n=n()) %>% nrow()

lambdas <- c(0.1, 0.5, 1, 2, 3, 5)
epochs <- 100

rmses <- numeric(length(lambdas))

for (lambda_idx in 1:length(lambdas)) {
  lambda = lambdas[lambda_idx]
  
  userFactMatrix <- initialiseUserFactorizationMatrix(nrow = USERS_REG, ncol = 2)
  movieFactMatrix <- initialiseMovieFactorizationMatrix(nrow = MOVIES_REG, ncol = 2)
  
  for (epoch in 1:epochs) {
    userFactMatrix <- update2ColUserFactorizationMatrix(edx_train_with_numbers, lambda, userFactMatrix, movieFactMatrix)
    movieFactMatrix <- update2ColMovieFactorizationMatrix(edx_train_with_numbers, lambda, userFactMatrix, movieFactMatrix)
  }
  
  # 3.1 COUNT RMSE FOR PARTICULAR LAMBDA
  edx_validation_with_estimations <- get2RankEstimations(edx_validation_with_numbers, userFactMatrix, movieFactMatrix)

  rmses[lambda_idx] <- RMSE(edx_validation_with_estimations$rating, edx_validation_with_estimations$rating_hat)
    
  print(c("LAMBDA - ", lambda, "RMSE - ", rmses[lambda_idx]))
}

# Define best lambda after regularization
BEST_LAMBDA <- lambdas[which.min(rmses)] 

#
# 4. DATA WRANGLING
#
edx_with_numbers <- setArrangedNumberToUsersAndMovies(edx)

validation_with_numbers <- applyUsersAndMoviesNumberFromOneDatasetToAnother(edx_with_numbers, validation)

#
# 5. RUN TRAINING ON THE HOLE TRAINING SET 
#
USERS_TRAIN <- edx_with_numbers %>% group_by(userId) %>% summarise(n=n()) %>% nrow()
MOVIES_TRAIN <- edx_with_numbers %>% group_by(movieId) %>% summarise(n=n()) %>% nrow()

userFactMatrixTrain <- initialiseUserFactorizationMatrix(nrow = USERS_TRAIN, ncol = 2)
movieFactMatrixTrain <- initialiseMovieFactorizationMatrix(nrow = MOVIES_TRAIN, ncol = 2)

for (epoch in 1:epochs) {
  userFactMatrixTrain <- update2ColUserFactorizationMatrix(edx_with_numbers, BEST_LAMBDA, userFactMatrixTrain, movieFactMatrixTrain)
  movieFactMatrixTrain <- update2ColMovieFactorizationMatrix(edx_with_numbers, BEST_LAMBDA, userFactMatrixTrain, movieFactMatrixTrain)
}

#
# 6. VALIDATION
#
validation_with_estimations <- get2RankEstimations(validation_with_numbers, userFactMatrixTrain, movieFactMatrixTrain)

validation_rmse <- RMSE(validation_with_estimations$rating, validation_with_estimations$rating_hat)

print(c('VALIDATION RMSE - ', validation_rmse))

