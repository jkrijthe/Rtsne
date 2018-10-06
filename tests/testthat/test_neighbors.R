context("Rtsne neighbor input")

set.seed(101)
# The original iris_matrix has a few tied distances, which alters the order of nearest neighbours.
# This then alters the order of addition when computing various statistics;
# which results in small rounding errors that are amplified across t-SNE iterations.
# Hence, I have to simulate a case where no ties are possible.
simdata <- matrix(rnorm(234*5), nrow=234)

test_that("Rtsne works correctly with nearest-neighbor input", {
    D <- as.matrix(dist(simdata))
    all.indices <- matrix(0L, nrow(simdata), 90) # 3 * perplexity
    all.distances <- matrix(0, nrow(simdata), 90)
    for (i in seq_len(nrow(simdata))) {
        current <- D[i,]
        by.dist <- setdiff(order(current), i)
        all.indices[i,] <- head(by.dist, ncol(all.indices))
        all.distances[i,] <- current[all.indices[i,]]
    }

    # The vptree involves a few R::runif calls, which alters the seed in the precomputed distance case.
    # This results in different random initialization of Y.
    # Thus, we need to supply a pre-computed Y as well.
    Y_in <- matrix(runif(nrow(simdata)*2), ncol=2)
    out <- Rtsne_neighbors(all.indices, all.distances, Y_init=Y_in, perplexity=30)
    blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=30)
    expect_equal(out$Y, blah$Y)

    # Trying again with a different number of neighbors.
    Y_in <- matrix(runif(nrow(simdata)*2), ncol=2)
    out <- Rtsne_neighbors(all.indices[,1:30], all.distances[,1:30], Y_init=Y_in, perplexity=10)
    blah <- Rtsne(D, is_distance=TRUE, Y_init=Y_in, perplexity=10)
    expect_equal(out$Y, blah$Y)
})

test_that("error conditions are correctly explored", {
    expect_error(Rtsne_neighbors("yay", matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "matrix")
    expect_error(Rtsne_neighbors(matrix(0L, 50, 5), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "same dimensions")
    expect_error(Rtsne_neighbors(matrix(0L, 50, 10), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "invalid indices")
    expect_error(Rtsne_neighbors(matrix(51L, 50, 10), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "invalid indices")
    expect_error(Rtsne_neighbors(matrix(NA_real_, 50, 10), matrix(0, 50, 10), Y_init=Y_in, perplexity=10), "invalid indices")
})
