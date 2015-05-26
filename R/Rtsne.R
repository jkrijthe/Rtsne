#' Barnes-Hut implementation of t-Distributed Stochastic Neighbor Embedding 
#' 
#' Wrapper for the C++ implementation of Barnes-Hut t-Distributed Stochastic Neighbor Embedding
#' 
#' After checking the correctness of the input, this function (optionally) does an initial reduction of the feature space using \code{\link{prcomp}}, before calling the C++ TSNE implementation. Since R's random number generator is used, use \code{\link{set.seed}} before the function call to get reproducible results.
#' 
#' Is \code{X} is a data.frame, it is transformed into a matrix using \code{\link{model.matrix}}. If \code{X} is a \code{\link{dist}} object, it is currently first expanded into a full distance matrix.
#' 
#' @param X matrix; Data matrix
#' @param dims integer; Output dimensionality (default: 2)
#' @param initial_dims integer; the number of dimensions that should be retained in the initial PCA step (default: 50)
#' @param perplexity numeric; Perplexity parameter
#' @param theta numeric; Speed/accuracy trade-off (increase for less accuracy), set to 0.0 for exact TSNE (default: 0.5)
#' @param check_duplicates logical; Checks whether duplicates are present. It is best to make sure there are no duplicates present and set this option to FALSE, especially for large datasets (default: TRUE)
#' @param pca logical; Whether an initial PCA step should be performed (default: TRUE)
#' @param max_iter integer; Maximum number of iterations (default: 1000)
#' @param verbose logical; Whether progress updates should be printed (default: FALSE)
#' @param ... Other arguments that can be passed to Rtsne
#' @param is_distance logical; Indicate whether X is a distance matrix (experimental, default: FALSE)
#' 
#' @return List with the following elements:
#' \item{Y}{Matrix containing the new representations for the objects}
#' \item{N}{Number of objects}
#' \item{origD}{Original Dimensionality before TSNE}
#' \item{perplexity}{See above}
#' \item{theta}{See above}
#' \item{costs}{The cost for every object after the final iteration}
#' \item{itercosts}{The total costs for all objects in every 50th + the last iteration}
#' 
#' @references L.J.P. van der Maaten. Barnes-Hut-SNE. In Proceedings of the International Conference on Learning Representations, 2013.
#' 
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' iris_matrix <- as.matrix(iris_unique[,1:4])
#' set.seed(42) # Set a seed if you want reproducible results
#' tsne_out <- Rtsne(iris_matrix) # Run TSNE
#' 
#' # Show the objects in the 2D tsne representation
#' plot(tsne_out$Y,col=iris_unique$Species)
#' 
#' # Using a dist object
#' tsne_out <- Rtsne(dist(iris_matrix))
#' plot(tsne_out$Y,col=iris_unique$Species)
#' 
#' @useDynLib Rtsne
#' @import Rcpp
#' 
#' @export
Rtsne <- function (X, ...) {
  UseMethod("Rtsne", X)
}

#' @describeIn Rtsne
#' @export
Rtsne.default <- function(X, dims=2, initial_dims=50, perplexity=30, theta=0.5, check_duplicates=TRUE, pca=TRUE,max_iter=1000,verbose=FALSE, is_distance=FALSE, ...) {
  
  if (!is.logical(is_distance)) { stop("is_distance should be a logical variable")}
  if (!is.numeric(theta) || (theta<0.0) || (theta>1.0) ) { stop("Incorrect theta.")}
  if (nrow(X) - 1 < 3 * perplexity) { stop("Perplexity is too large.")}
  if (!is.matrix(X)) { stop("Input X is not a matrix")}
  if (!(max_iter>0)) { stop("Incorrect number of iterations.")}
  if (is_distance & !(is.matrix(X) & (nrow(X)==ncol(X)))) { stop("Input is not an accepted distance matrix") }
  
  is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol
  if (!is.wholenumber(initial_dims) || initial_dims<=0) { stop("Incorrect initial dimensionality.")}
  if (check_duplicates & !is_distance){
    if (any(duplicated(X))) { stop("Remove duplicates before running TSNE.") }
  }
  
  # Apply PCA
  if (pca & !is_distance) {
    pca_result <- prcomp(X,retx=TRUE)
    X <- pca_result$x[,1:min(initial_dims,ncol(pca_result$x))]
  }
  # Compute Squared distance if we are using exact TSNE
  if (is_distance & theta==0.0) {
    X <- X^2
  }
  
  Rtsne_cpp(X, dims, perplexity, theta,verbose, max_iter, is_distance)
}

#' @describeIn Rtsne
#' @export
Rtsne.dist <- function(X,...,is_distance=TRUE) {
  X <- as.matrix(X)
  Rtsne(X, ..., is_distance=is_distance)
}

#' @describeIn Rtsne
#' @export
Rtsne.data.frame <- function(X,...) {
  X <- model.matrix(~.-1,X)
  Rtsne(X, ...)
}
