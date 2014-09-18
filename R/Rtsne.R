#' Barnes-Hut implementation of t-Distributed Stochastic Neighbor Embedding 
#' 
#' Wrapper for the C++ implementation of Barnes-Hut t-Distributed Stochastic Neighbor Embedding
#' 
#' After checking the correctness of the input, this function (optionally) does an initial reduction of the feature space using \code{\link{prcomp}}, before calling the C++ TSNE implementation. Since R's random number generator is used, use \code{\link{set.seed}} before the function call to get reproducable results.
#' 
#' @param X matrix; Data matrix
#' @param dims integer; Output dimensionality (default: 2)
#' @param initial_dims integer; the number of dimensions that should be retained in the initial PCA step (default: 50)
#' @param perplexity numeric; Perplexity parameter
#' @param theta numeric; Speed/accuracy trade-off (increase for less accuracy) (default: 0.5)
#' @param check_duplicates logical; Checks whether duplicates are present. It is best to make sure there are no duplicates present and set this option to FALSE, especially for large datasets (default: TRUE)
#' @param pca logical; Whether an initial PCA step should be performed (default: TRUE)
#' 
#' @return List with the following elements:
#' \item{Y}{Matrix containing the new representations for the objects}
#' \item{N}{Number of objects}
#' \item{origD}{Original Dimensionality before TSNE}
#' \item{perplexity}{See above}
#' \item{theta}{See above}
#' 
#' @references L.J.P. van der Maaten. Barnes-Hut-SNE. In Proceedings of the International Conference on Learning Representations, 2013.
#' 
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' tsne_out <- Rtsne(as.matrix(iris_unique[,1:4])) # Run TSNE
#' plot(tsne_out$Y,col=iris$Species) # Plot the result
#' 
#' @useDynLib Rtsne
#' @import Rcpp
#' 
#' @export
Rtsne<-function(X, dims=2, initial_dims=50, perplexity=30, theta=0.5, check_duplicates=TRUE, pca=TRUE) {
  if (!is.numeric(theta) | (theta<=0.0) | (theta>1.0) ) { stop("Incorrect theta.")}
  if (nrow(X) - 1 < 3 * perplexity) { stop("Perplexity is too large.")}
  if (!is.matrix(X)) { stop("Input X is not a matrix")}
  
  is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol
  if (!is.wholenumber(initial_dims) | initial_dims<=0) { stop("Incorrect initial dimensionality.")}
  if (check_duplicates){
    if (any(duplicated(X))) { stop("Remove duplicates before running TSNE.") }
  }
  
  # Apply PCA
  if (pca) {
    X <- prcomp(X,retx=TRUE)$x[,1:min(initial_dims,ncol(X))]
  }
  
  Rtsne_cpp(X,dims,perplexity,theta)
}