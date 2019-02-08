context("Rtsne neighbor input")

set.seed(101)
# The original iris_matrix has a few tied distances, which alters the order of nearest neighbours.
# This then alters the order of addition when computing various statistics;
# which results in small rounding errors that are amplified across t-SNE iterations.
# Hence, I have to simulate a case where no ties are possible.
simdata <- matrix(rnorm(234*5), nrow=234)

NNFUN <- function(D, K) 
# A quick reference function for computing NNs, to avoid depending on other packages.
{
    all.indices <- matrix(0L, nrow(D), K)
    all.distances <- matrix(0, nrow(D), K)
    for (i in seq_len(nrow(D))) {
        current <- D[i,]
        by.dist <- setdiff(order(current), i)
        all.indices[i,] <- head(by.dist, ncol(all.indices))
        all.distances[i,] <- current[all.indices[i,]]
    }
    list(index=all.indices, distance=all.distances)
}

test_that("Rtsne with nearest-neighbor input compares to distance matrix input", {
    D <- as.matrix(dist(simdata))
    out <- NNFUN(D, 90) # 3 * perplexity
    all.indices <- out$index
    all.distances <- out$distance

    # The vptree involves a few R::runif calls, which alters the seed in the precomputed distance case.
    # This results in different random initialization of Y.
    # Thus, we need to supply a pre-computed Y as well.
    Y_in <- matrix(runif(nrow(simdata)*2), ncol=2)
    out <- Rtsne_neighbors(all.indices, all.distances, Y_init=Y_in, perplexity=30, max_iter=1)
    blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=30, max_iter=1)
    expect_equal(out$Y, blah$Y)

    # Trying again with a different number of neighbors.
    Y_in <- matrix(runif(nrow(simdata)*2), ncol=2)
    out <- Rtsne_neighbors(all.indices[,1:30], all.distances[,1:30], Y_init=Y_in, perplexity=10)
    blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=10)
    expect_equal(out$Y, blah$Y)
})

test_that("Rtsne with nearest-neighbor input behaves upon normalization", {
    D <- as.matrix(dist(normalize_input(simdata)))
    out <- NNFUN(D, 90) # 3 * perplexity
    all.indices <- out$index
    all.distances <- out$distance

    # The vptree involves a few R::runif calls, which alters the seed in the precomputed distance case.
    # This results in different random initialization of Y.
    # Thus, we need to supply a pre-computed Y as well.
    Y_in <- matrix(runif(nrow(simdata)*2), ncol=2)
    out <- Rtsne_neighbors(all.indices, all.distances, Y_init=Y_in, perplexity=30, max_iter=1)
    blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=30, max_iter=1)
    expect_equal(out$Y, blah$Y)

    # Trying again with a different number of neighbors.
    Y_in <- matrix(runif(nrow(simdata)*2), ncol=2)
    out <- Rtsne_neighbors(all.indices[,1:30], all.distances[,1:30], Y_init=Y_in, perplexity=10)
    blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=10)
    expect_equal(out$Y, blah$Y)
})

test_that("Rtsne with nearest-neighbor input gives no errors for no_dims 1 and 3", {
  D <- as.matrix(dist(simdata))
  out <- NNFUN(D, 90) # 3 * perplexity
  all.indices <- out$index
  all.distances <- out$distance
  
  # The vptree involves a few R::runif calls, which alters the seed in the precomputed distance case.
  # This results in different random initialization of Y.
  # Thus, we need to supply a pre-computed Y as well.
  Y_in <- matrix(runif(nrow(simdata)*1), ncol=1)
  out <- Rtsne_neighbors(all.indices, all.distances, Y_init=Y_in, perplexity=30, max_iter=1, dims = 1)
  blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=30, max_iter=1,dims = 1)
  expect_equal(out$Y, blah$Y)
  
  Y_in <- matrix(runif(nrow(simdata)*3), ncol=3)
  out <- Rtsne_neighbors(all.indices, all.distances, Y_init=Y_in, perplexity=30, max_iter=1, dims = 3)
  blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=30, max_iter=1,dims = 3)
  expect_equal(out$Y, blah$Y)
})

test_that("error conditions are correctly explored", {
    expect_error(Rtsne_neighbors("yay", matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "matrix")
    expect_error(Rtsne_neighbors(matrix(0L, 50, 5), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "same dimensions")
    expect_error(Rtsne_neighbors(matrix(0L, 50, 10), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "invalid indices")
    expect_error(Rtsne_neighbors(matrix(51L, 50, 10), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "invalid indices")
    expect_error(Rtsne_neighbors(matrix(NA_real_, 50, 10), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "invalid indices")
})
