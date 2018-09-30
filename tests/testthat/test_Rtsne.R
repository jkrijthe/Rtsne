context("Rtsne main function")

# Prepare iris dataset
iris_unique <- unique(iris) # Remove duplicates
Xscale <- scale(as.matrix(iris_unique[,1:4]), 
                center = TRUE, scale=FALSE)
Xscale <- scale(Xscale,
                center=FALSE, 
                scale=rep(max(abs(Xscale)),4))
distmat <- as.matrix(dist(Xscale))

# Run models to compare to
set.seed(50)
tsne_out_matrix <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, is_distance = FALSE,theta=0.0,max_iter=200,pca=FALSE)
set.seed(50)
tsne_out_matrix_bh <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=200)

test_that("Manual distance calculation equals C++ distance calculation", {
  
  # Exact
  # Note: only the same in the first few iterations
  set.seed(50)
  tsne_matrix <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, is_distance = FALSE,theta=0.0,max_iter=5,pca=FALSE)
  set.seed(50)
  tsne_dist <- Rtsne(distmat, verbose=FALSE, is_distance = TRUE,theta = 0.0,max_iter=5)
  expect_equal(tsne_matrix$Y,tsne_dist$Y,tolerance=10e-5)
  
  # Inexact
  set.seed(50)
  tsne_matrix <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, is_distance = FALSE,theta=0.1,max_iter=5,pca=FALSE)
  #plot(tsne_matrix$Y)
  set.seed(50)
  tsne_dist <- Rtsne(distmat, verbose=FALSE, is_distance = TRUE, theta = 0.1, max_iter=5,pca=FALSE)
  #plot(tsne_dist$Y)
  expect_equal(tsne_matrix$Y,tsne_dist$Y,tolerance=10e-5)
})

test_that("Accepts dist", {
  # Exact
  set.seed(50)
  tsne_out_dist_matrix <- Rtsne(distmat, is_distance = TRUE, theta=0.0, max_iter=1000)
  set.seed(50)
  tsne_out_dist <- Rtsne(dist(Xscale),theta=0.0,max_iter=1000)
  expect_equal(tsne_out_dist$Y,tsne_out_dist_matrix$Y)
  
  # Inexact
  set.seed(50)
  tsne_out_dist_matrix <- Rtsne(distmat, is_distance = TRUE, theta=0.1, max_iter=1000)
  set.seed(50)
  tsne_out_dist <- Rtsne(dist(Xscale), theta=0.1, max_iter=1000)
  expect_equal(tsne_out_dist$Y,tsne_out_dist_matrix$Y)
})

test_that("Accepts data.frame", {
  #Exact
  set.seed(50)
  tsne_out_df <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.0,pca=FALSE,max_iter=200,num_threads=1)
  expect_equal(tsne_out_matrix$Y,tsne_out_df$Y)
  
  #Inexact
  set.seed(50)
  tsne_out_df <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=200,num_threads=1)
  expect_equal(tsne_out_matrix_bh$Y,tsne_out_df$Y)
})

test_that("OpenMP with different threads returns same result",{
  iter_equal <- 20
  
  set.seed(50)
  tsne_out_df1 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,
                       theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=1)
  set.seed(50)
  tsne_out_df2 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,
                       theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=2)
  set.seed(50)
  tsne_out_df3 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,
                       theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=3)
  expect_equal(tsne_out_df1$Y,tsne_out_df2$Y)
  expect_equal(tsne_out_df2$Y,tsne_out_df3$Y)
})

# test_that("Continuing from initialization gives approximately the same result as direct run", {
#   iter_equal <- 20
#   
#   #Exact
#   set.seed(50)
#   tsne_out_full <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.0,pca=FALSE,max_iter=500)
#   set.seed(50)
#   tsne_out_part1 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.0,pca=FALSE,max_iter=300)
#   set.seed(50)
#   tsne_out_part2 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.0,pca=FALSE,max_iter=200,Y_init=tsne_out_part1$Y)
#   expect_equal(tsne_out_full$Y,tsne_out_part2$Y, tolerance = .001,scale=1)
# 
#   #Inexact
#   set.seed(50)
#   tsne_out_full <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=1000)
#   set.seed(50)
#   tsne_out_part1 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=750)
#   set.seed(50)
#   tsne_out_part2 <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=250,Y_init=tsne_out_part1$Y)
#   expect_equal(tsne_out_full$Y,tsne_out_part2$Y, tolerance = .001,scale=1)
# })

test_that("partial_pca FALSE and TRUE give similar results", {
  iter_equal <- 5
  
  set.seed(42)
  fat_data <- rbind(sapply(runif(200,-1,1), function(x) rnorm(200,x)),
                    sapply(runif(200,-1,1), function(x) rnorm(200,x)))
  
  set.seed(42)
  tsne_out_prcomp <- Rtsne(fat_data, max_iter = iter_equal)
  
  set.seed(42)
  tsne_out_irlba <- Rtsne(fat_data, partial_pca = T, max_iter = iter_equal)

  # Sign of principal components are arbitrary so even with same seed tSNE coordinates are not the same
  expect_equal(tsne_out_prcomp$costs, tsne_out_irlba$costs, tolerance = .01, scale = 1)
})

