if(!require(bigmemory)) install.packages("bigmemory", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")

library(bigmemory)
library(matrixStats)

EXTRA_SMALL <- 1e-16

TRAINING_MODE <- 'training'
REGULARIZATION_MODE <- 'regularization'
# Initialise Gaussian mixture
# with given matrix of ratings and number of clusters
init_mixture <- function(user_movie_matrix, clusters) {
  user_amount <- nrow(user_movie_matrix)
  
  # Initial probabilities
  probs <- rep(1/clusters, clusters)
  
  mus <- user_movie_matrix[sample(1:user_amount, clusters, replace = F),]
  
  # Initial deviation
  vars <- sapply(1:clusters, function (cluster) {
    mean((sweep(user_movie_matrix[,], 2, mus[cluster,])) ** 2)
  })
  
  list(mus, vars, probs)
}

get_log_vector_norm_density <- function(vector, mus, var) {
   ((-norm(vector - mus, type = '2') ** 2) / (2 * var)) - (length(vector) / 2) * log(2 * pi * var)
}

prob_of_user_in_clusters <- function(user_ratings, mixture) {
  sapply(1:length(mixture[[2]]), function(cluster_i, user_ratings, mixture) {
    mus <- mixture[[1]]
    vars <- mixture[[2]]
    probs <- mixture[[3]]
    
    c_u <- which(user_ratings > 0)
    given_user_ratings <- user_ratings[c_u]
    mu_for_give_ratings_in_cluster <- mus[cluster_i,c_u]

    cluster_prob <- probs[cluster_i]
    log_norm_prob_of_user_in_cluster <- get_log_vector_norm_density(given_user_ratings, mu_for_give_ratings_in_cluster, vars[cluster_i])
    
    log(cluster_prob + EXTRA_SMALL) + log_norm_prob_of_user_in_cluster
  }, user_ratings, mixture)
}

# Sets rating to given matrix[user_number, movie_number]
set_rating_to_matrix <- function(data_row, dataset, matrix) {
  print(data_row/nrow(dataset))
  matrix[dataset[data_row]$user_number, dataset[data_row]$movie_number] <- dataset[data_row]$rating
}

set_zeros <- function(data_row, dataset, matrix) {
  print(data_row/nrow(dataset))
  matrix[dataset[data_row]$user_number, dataset[data_row]$movie_number] <- 0
}

# Converts raitings dataset to matrix with shape users:movies 
dataset_to_matrix <- function(dataset) {
  data_matrix <- big.matrix(nrow = max(dataset$user_number), ncol = max(dataset$movie_number), init = 0, type = 'float')

  lapply(1:nrow(dataset), set_rating_to_matrix, dataset = dataset, matrix = data_matrix)
  
  data_matrix
}

# Returns data file path depending on mode
get_file_path <- function(mode = TRAINING_MODE) {
  if (mode == TRAINING_MODE) {
    path <- 'generated_data/data_incomplete'
  } else {
    path <- 'generated_data/data_regul_incomplete'
  }
  
  path
}


# Returns data_matrix depending on mode
get_data_matrix <- function(mode = TRAINING_MODE) {
  file_path <- get_file_path(mode)
  
  if (file.exists(file_path)) {
    data_matrix <- read.big.matrix(file_path, type = 'float')
  } else {
    
    
    data_matrix <- dataset_to_matrix(edx_with_numbers)
    
    if (mode == REGULARIZATION_MODE) {
      data_matrix <- set_zeros_from_dataset_to_matrix(edx_validation_with_numbers, data_matrix)
    }
    write.big.matrix(data_matrix, file = file_path, row.names = F, col.names = F)
  }
  
  data_matrix
}

set_zeros_from_dataset_to_matrix <- function(dataset, matrix) {
  sapply(1:nrow(dataset), set_zeros, dataset, matrix)
  
  matrix
}

# Counting posterior probabilities
# for each point to be related to each cluster
# and log_likelihood for current mixture
e_step <- function(user_movie_matrix, mixture) {
  user_amount <- nrow(user_movie_matrix)
  K <- length(mixture[[2]])
  
  log_user_probs <- sapply(1:user_amount, function(user_idx, user_movie_matrix, mixture) {
    prob_of_user_in_clusters(user_movie_matrix[user_idx,], mixture)
  }, user_movie_matrix, mixture)
  
  log_sum_all_clusters <- sapply(1:user_amount, function(user_idx, log_user_probs) {
    logSumExp(log_user_probs[,user_idx])
  }, log_user_probs)
  
  
  log_posteriors <- sweep(log_user_probs, 2, log_sum_all_clusters)
  
  log_likelihood <- sum(log_sum_all_clusters)
  
  list(exp(t(log_posteriors)), log_likelihood)
}

m_step <- function(user_movie_matrix, mixture, posteriors, min_variance = 0.25) {
  user_amount <- nrow(user_movie_matrix)
  movie_amount <- ncol(user_movie_matrix)
  clusters <- length(mixture[[2]])
  
  old_mus <- mixture[[1]]
  
  sum_posteriors_by_cluster <- colSums(posteriors)
  

  mus_hat <- t(sapply(1:clusters, function(cluster, mixture, posteriors, user_movie_matrix, old_mus) {
    user_probs_by_cluster <- posteriors[,cluster]
    
    sapply(1:movie_amount, function(movie_idx, user_movie_matrix, user_probs_by_cluster, old_mus, cluster) {
      denom <- sum(user_probs_by_cluster[which(user_movie_matrix[,movie_idx] != 0)])
      
      if (denom >= 1) {
        result <- sum(user_probs_by_cluster * user_movie_matrix[,movie_idx])/denom
      } else {
        result <- old_mus[cluster, movie_idx]
      }
      
      result
    }, user_movie_matrix, user_probs_by_cluster, old_mus, cluster)
  }, mixture, posteriors, user_movie_matrix, old_mus))
  
  user_given_ratings_amount <- rowSums(user_movie_matrix[,] != 0)
  
  vars_hat <- sapply(1:clusters, function(cluster, user_movie_matrix, user_given_ratings_amount, posteriors, mus_hat) {
    user_probs_by_cluster <- posteriors[,cluster]
    user_amount <- length(user_probs_by_cluster)
    
    denom <- sum(user_probs_by_cluster * user_given_ratings_amount)
    
    sum(sapply(1:user_amount, function(user_idx, user_movie_matrix, user_probs_by_cluster, cluster, mus_hat) {
      user_given_ratings_indexes <- which(user_movie_matrix[user_idx,] != 0)
      
      user_probs_by_cluster[user_idx] * norm(user_movie_matrix[user_idx, user_given_ratings_indexes] - mus_hat[cluster, user_given_ratings_indexes], type = '2') ** 2
    }, user_movie_matrix, user_probs_by_cluster, cluster, mus_hat))/denom
  }, user_movie_matrix, user_given_ratings_amount, posteriors, mus_hat)
  
  vars_hat[which(vars_hat < min_variance)] <- min_variance
  
  probs_hat <- sum_posteriors_by_cluster/user_amount
  
  list(mus_hat, vars_hat, probs_hat)
}

# Fills ratings for one user
fill_user_ratings <- function(user_ratings, mus, user_posteriors) {
  empty_ratings_indexes <- which(user_ratings == 0)
  
  mus_for_estimation <- mus[,empty_ratings_indexes]

  estimated_ratings <- rowSums(t(mus_for_estimation * user_posteriors))
  
  user_ratings[empty_ratings_indexes] <- estimated_ratings
  
  user_ratings
}


# Fill hole matrix with estimated ratings
fill_matrix <- function(user_movie_matrix, mixture, posteriors) {
  user_amount <- nrow(user_movie_matrix)
  
  mus <- mixture[[1]]
  
  filled_matrix <- sapply(1:user_amount, function(row, user_movie_matrix, mixture, posteriors) {
    print(row)
    fill_user_ratings(user_movie_matrix[row,], mus, posteriors[row,])
  }, user_movie_matrix, mixture, posteriors)
  
  t(filled_matrix)
}

K <- seq(8,16,2)
likelihoods <- vector(mode = 'numeric', length(K))
EPSILON <- 10e-6
data_matrix <- get_data_matrix(TRAINING_MODE)
regul_matrix <- data_matrix[1:5000, 1:3000]

for (i in 1:length(K)) {
  clusters <- K[i]
  
  mixture <- init_mixture(regul_matrix, clusters)
  
  e_step_result <- e_step(regul_matrix, mixture)
  posteriors <- e_step_result[[1]]
  new_log_likelihood <- e_step_result[[2]]
  print(new_log_likelihood)
  
  mixture <- m_step(regul_matrix, mixture, posteriors)
  
  differ <- Inf
  
  while(differ >= abs(new_log_likelihood) * EPSILON) {
    old_log_likelihood <- new_log_likelihood
    
    e_step_result <- e_step(regul_matrix, mixture)
    posteriors <- e_step_result[[1]]
    new_log_likelihood <- e_step_result[[2]]
    print(new_log_likelihood)
    
    differ <- new_log_likelihood - old_log_likelihood
    
    if (differ <= abs(new_log_likelihood) * EPSILON) {
      break
    }
    
    mixture <- m_step(regul_matrix, mixture, posteriors)
  }
  likelihoods[i] <- new_log_likelihood
}

BEST_K <- K[which.max(likelihoods)]

mixture <- init_mixture(data_matrix, BEST_K)

e_step_result <- e_step(data_matrix, mixture)
posteriors <- e_step_result[[1]]
new_log_likelihood <- e_step_result[[2]]
print(new_log_likelihood)

mixture <- m_step(data_matrix, mixture, posteriors)

differ <- Inf

while(differ >= abs(new_log_likelihood) * EPSILON) {
  old_log_likelihood <- new_log_likelihood
  
  e_step_result <- e_step(data_matrix, mixture)
  posteriors <- e_step_result[[1]]
  new_log_likelihood <- e_step_result[[2]]
  print(new_log_likelihood)
  
  differ <- new_log_likelihood - old_log_likelihood
  
  if (differ <= abs(new_log_likelihood) * EPSILON) {
    break
  }
  
  mixture <- m_step(data_matrix, mixture, posteriors)
}

print('Completed')

filled_matrix <- fill_matrix(data_matrix, mixture, posteriors)

rating_hats <- sapply(1:nrow(validation_with_numbers), function(row, validation_with_numbers, filled_matrix) {
  user_number <- validation_with_numbers[row]$user_number
  movie_number <- validation_with_numbers[row]$movie_number
  
  filled_matrix[user_number, movie_number]
}, validation_with_numbers, filled_matrix)

RMSE(validation_with_numbers$rating, rating_hats)

