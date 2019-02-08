context("Rtsne main function")

# Prepare iris dataset
iris_unique <- unique(iris) # Remove duplicates
Xscale<-normalize_input(as.matrix(iris_unique[,1:4]))
distmat <- as.matrix(dist(Xscale))

# Run models to compare to
iter_equal <- 500

test_that("Scaling gives the expected result", {
  Xscale2 <- scale(as.matrix(iris_unique[,1:4]), 
                  center = TRUE, scale=FALSE)
  Xscale2 <- scale(Xscale,
                  center=FALSE, 
                  scale=rep(max(abs(Xscale)),4))
  expect_equivalent(Xscale,Xscale2)
})

test_that("Manual distance calculation equals C++ distance calculation", {
  
  # Does not work on 32 bit windows
  skip_on_cran()
  
  # Exact
  set.seed(50)
  tsne_matrix <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, 
                       is_distance = FALSE,theta=0.0,max_iter=iter_equal,
                       pca=FALSE, normalize=TRUE)
  set.seed(50)
  tsne_dist <- Rtsne(distmat, verbose=FALSE, is_distance = TRUE,
                     theta = 0.0, max_iter=iter_equal)
  expect_equal(tsne_matrix$Y,tsne_dist$Y)
  
  # Inexact
  set.seed(50)
  tsne_matrix <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, is_distance = FALSE,
                       theta=0.1,max_iter=iter_equal,pca=FALSE)
  set.seed(50)
  tsne_dist <- Rtsne(distmat, verbose=FALSE, is_distance = TRUE, theta = 0.1, max_iter=iter_equal,
                     pca=FALSE)
  expect_equal(tsne_matrix$Y,tsne_dist$Y)
})

test_that("Accepts dist", {
  
  # Exact
  set.seed(50)
  tsne_out_dist_matrix <- Rtsne(distmat, is_distance = TRUE, theta=0.0, max_iter=iter_equal)
  set.seed(50)
  tsne_out_dist <- Rtsne(dist(Xscale),theta=0.0,max_iter=iter_equal)
  expect_equal(tsne_out_dist$Y,tsne_out_dist_matrix$Y)
  
  # Inexact
  set.seed(50)
  tsne_out_dist_matrix <- Rtsne(distmat, is_distance = TRUE, theta=0.1, max_iter=iter_equal)
  set.seed(50)
  tsne_out_dist <- Rtsne(dist(Xscale), theta=0.1, max_iter=iter_equal)
  expect_equal(tsne_out_dist$Y,tsne_out_dist_matrix$Y)
})

test_that("Accepts data.frame", {
  
  # Exact
  set.seed(50)
  tsne_out_matrix <- Rtsne(as.matrix(iris_unique[,1:4]),dims=1,verbose=FALSE, is_distance = FALSE,theta=0.0,max_iter=iter_equal,pca=FALSE)
  set.seed(50)
  tsne_out_df <- Rtsne(iris_unique[,1:4],dims=1,verbose=FALSE, is_distance = FALSE,theta=0.0,pca=FALSE,max_iter=iter_equal,num_threads=1)
  expect_equal(tsne_out_matrix$Y,tsne_out_df$Y)
  
  # Inexact
  set.seed(50)
  tsne_out_matrix_bh <- Rtsne(as.matrix(iris_unique[,1:4]),verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=iter_equal)
  set.seed(50)
  tsne_out_df <- Rtsne(iris_unique[,1:4],verbose=FALSE, is_distance = FALSE,theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=1)
  expect_equal(tsne_out_matrix_bh$Y,tsne_out_df$Y)
})

test_that("OpenMP with different threads returns same result",{
  
  # Does not work on windows
  skip_on_cran()
  
  set.seed(50)
  tsne_out_df1 <- Rtsne(iris_unique[,1:4],dims=3,verbose=FALSE, is_distance = FALSE,
                       theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=1)
  set.seed(50)
  tsne_out_df2 <- Rtsne(iris_unique[,1:4],dims=3,verbose=FALSE, is_distance = FALSE,
                       theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=2)
  set.seed(50)
  tsne_out_df3 <- Rtsne(iris_unique[,1:4],dims=3,verbose=FALSE, is_distance = FALSE,
                       theta=0.1,pca=FALSE,max_iter=iter_equal,num_threads=3)
  expect_equal(tsne_out_df1$Y,tsne_out_df2$Y)
  expect_equal(tsne_out_df2$Y,tsne_out_df3$Y)
})

test_that("Continuing from initialization gives approximately the same result as direct run", {

  # Does not work exactly due to resetting of "gains".
  iter_equal <- 1000
  extra_iter <- 200
  
  #Exact
  set.seed(50)
  tsne_out_full <- Rtsne(iris_unique[,1:4],
                         perplexity=3,theta=0.0,pca=FALSE,
                         max_iter=iter_equal,final_momentum = 0)
  set.seed(50)
  tsne_out_part1 <- Rtsne(iris_unique[,1:4],
                          perplexity=3,theta=0.0,pca=FALSE,
                          max_iter=iter_equal-extra_iter,final_momentum = 0)
  tsne_out_part2 <- Rtsne(iris_unique[,1:4],
                          perplexity=3,theta=0.0,pca=FALSE,
                          max_iter=extra_iter,Y_init=tsne_out_part1$Y,final_momentum = 0)
  expect_equivalent(dist(tsne_out_full$Y),dist(tsne_out_part2$Y),tolerance=0.01)

  #Inexact
  set.seed(50)
  tsne_out_full <- Rtsne(iris_unique[,1:4],final_momentum=0,theta=0.1,pca=FALSE,max_iter=iter_equal)
  set.seed(50)
  tsne_out_part1 <- Rtsne(iris_unique[,1:4],final_momentum=0,theta=0.1,pca=FALSE,max_iter=iter_equal-extra_iter)
  set.seed(50)
  tsne_out_part2 <- Rtsne(iris_unique[,1:4],final_momentum=0,theta=0.1,pca=FALSE,max_iter=extra_iter,Y_init=tsne_out_part1$Y)
  expect_equivalent(dist(tsne_out_full$Y),dist(tsne_out_part2$Y),tolerance=0.01)
})

test_that("partial_pca FALSE and TRUE give similar results", {
  
  # Only first few iterations
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

test_that("Error conditions", {
  expect_error(Rtsne("test", matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "matrix")
  expect_error(Rtsne(distmat,is_distance = 3),"logical")
  expect_error(Rtsne(matrix(0,2,3),is_distance = TRUE),"Input")
  expect_error(Rtsne(matrix(0,100,3)),"duplicates")
  expect_error(Rtsne(matrix(0,2,3),pca_center = 2),"TRUE")
  expect_error(Rtsne(matrix(0,2,3),initial_dims=1.3),"dimensionality")
  expect_error(Rtsne(matrix(0,2,3),dims=4),"dims")
  expect_error(Rtsne(matrix(0,2,3),max_iter=1.5),"should")
  
  expect_error(Rtsne(matrix(0,2,3),Y_init=matrix(0,2,1)),"incorrect format")
  expect_error(Rtsne(matrix(0,2,3),perplexity = 0),"positive")
  expect_error(Rtsne(matrix(0,2,3),theta = -0.1),"lie")
  expect_error(Rtsne(matrix(0,2,3),theta = 1.001),"lie")
  expect_error(Rtsne(matrix(0,2,3),stop_lying_iter = -1),"positive")
  expect_error(Rtsne(matrix(0,2,3),mom_switch_iter = -1),"positive")
  expect_error(Rtsne(matrix(0,2,3),momentum = -0.1),"positive")
  expect_error(Rtsne(matrix(0,2,3),final_momentum = -0.1),"positive")
  expect_error(Rtsne(matrix(0,2,3),eta = 0.0),"positive")
  expect_error(Rtsne(matrix(0,2,3),exaggeration_factor = 0.0),"positive")
  expect_error(Rtsne(matrix(0,2,3)),"perplexity is too large")
})

test_that("Verbose option", {
  expect_output(Rtsne(iris_unique[,1:4],pca=TRUE,verbose=TRUE,max_iter=150),"Fitting performed")
})
