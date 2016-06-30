#' Barnes-Hut implementation of t-Distributed Stochastic Neighbor Embedding 
#' 
#' Wrapper for the C++ implementation of Barnes-Hut t-Distributed Stochastic Neighbor Embedding. t-SNE is a method for constructing a low dimensional embedding of high-dimensional data, distances or similarities. Exact t-SNE can be computed by setting theta=0.0. 
#' 
#' Given a distance matrix \eqn{D} between input objects (which by default, is the euclidean distances between two objects), we calculate a similarity score in the original space p_ij. \deqn{ p_{j | i} = \frac{\exp(-\|D_{ij}\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|D_{ij}\|^2 / 2 \sigma_i^2)} } which is then symmetrized using: \deqn{ p_{i j}=\frac{p_{j|i} + p_{i|j}}{2n}}. The \eqn{\sigma} for each object is chosen in such a way that the perplexity of p_{j|i} has a value that is close to the user defined perplexity. This value effectively controls how many nearest neighbours are taken into account when constructing the embedding in the low-dimensional space.
#' For the low-dimensional space we use the Cauchy distribution (t-distribution with one degree of freedom) as the distribution of the distances to neighbouring objects:
#' \deqn{ q_{i j} = \frac{(1+ \| y_i-y_j\|^2)^{-1}}{\sum_{k \neq l} 1+ \| y_k-y_l\|^2)^{-1}}}. 
#' By changing the location of the objects y in the embedding to minimize the Kullback-Leibler divergence between these two distributions \eqn{ q_{i j}} and \eqn{ p_{i j}}, we create a map that focusses on small-scale structure, due to the assymetry of the KL-divergence. The t-distribution is chosen to avoid the crowding problem: in the original high dimensional space, there are potentially many equidistant objects with moderate distance from a particular object, more than can be accounted for in the low dimensional representation. The t-distribution makes sure that these objects are more spread out in the new representation.
#' 
#' For larger datasets, a problem with the a simple gradient descent to minimize the Kullback-Leibler divergence is the computational complexity of each gradient step (which is \eqn{O(n^2)}). The Barnes-Hut implementation of the algorithm attempts to mitigate this problem using two tricks: (1) approximating small similarities by 0 in the \eqn{p_{ij}} distribution, where the non-zero entries are computed by finding 3*perplexity nearest neighbours using an efficient tree search. (2) Using the Barnes-Hut algorithm in the computation of the gradient which approximates large distance similarities using a quadtree. This approximation is controlled by the \code{theta} parameter, with smaller values leading to more exact approximations. When \code{theta=0.0}, the implementation uses a standard t-SNE implementation. The Barnes-Hut approximation leads to a \eqn{O(n log(n))} computational complexity for each iteration.
#' 
#' During the minimization of the KL-divergence, the implementation uses a trick known as early exaggeration, which multiplies the \eqn{p_{ij}}'s by 12 during the first 250 iterations. This leads to tighter clustering and more distance between clusters of objects. This early exaggeration is not used when the user gives an initialization of the objects in the embedding by setting \code{Y_init}. During the early exaggeration phase, a momentum term of 0.5 is used while this is changed to 0.8 after the first 250 iterations.
#' 
#' After checking the correctness of the input, the \code{Rtsne} function (optionally) does an initial reduction of the feature space using \code{\link{prcomp}}, before calling the C++ TSNE implementation. Since R's random number generator is used, use \code{\link{set.seed}} before the function call to get reproducible results.
#' 
#' If \code{X} is a data.frame, it is transformed into a matrix using \code{\link{model.matrix}}. If \code{X} is a \code{\link{dist}} object, it is currently first expanded into a full distance matrix.
#' 
#' @param X matrix; Data matrix
#' @param dims integer; Output dimensionality (default: 2)
#' @param initial_dims integer; the number of dimensions that should be retained in the initial PCA step (default: 50)
#' @param perplexity numeric; Perplexity parameter
#' @param theta numeric; Speed/accuracy trade-off (increase for less accuracy), set to 0.0 for exact TSNE (default: 0.5)
#' @param check_duplicates logical; Checks whether duplicates are present. It is best to make sure there are no duplicates present and set this option to FALSE, especially for large datasets (default: TRUE)
#' @param pca logical; Whether an initial PCA step should be performed (default: TRUE)
#' @param max_iter integer; Number of iterations (default: 1000)
#' @param verbose logical; Whether progress updates should be printed (default: FALSE)
#' @param ... Other arguments that can be passed to Rtsne
#' @param is_distance logical; Indicate whether X is a distance matrix (experimental, default: FALSE)
#' @param Y_init matrix; Initial locations of the objects. If NULL, random initialization will be used (default: NULL). Note that when using this, the initial stage with exaggerated perplexity values and a larger momentum term will be skipped.
#' 
#' @return List with the following elements:
#' \item{Y}{Matrix containing the new representations for the objects}
#' \item{N}{Number of objects}
#' \item{origD}{Original Dimensionality before TSNE}
#' \item{perplexity}{See above}
#' \item{theta}{See above}
#' \item{costs}{The cost for every object after the final iteration}
#' \item{itercosts}{The total costs (KL-divergence) for all objects in every 50th + the last iteration}
#' 
#' @references Maaten, L. Van Der, 2014. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research, 15, p.3221-3245.
#' @references van der Maaten, L.J.P. & Hinton, G.E., 2008. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research, 9, pp.2579-2605.
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
#' # Use a given initialization of the locations of the points
#' tsne_part1 <- Rtsne(iris_unique[,1:4], theta=0.0, pca=FALSE,max_iter=350)
#' tsne_part2 <- Rtsne(iris_unique[,1:4], theta=0.0, pca=FALSE, max_iter=150,Y_init=tsne_part1$Y)
#' @useDynLib Rtsne
#' @import Rcpp
#' @importFrom stats model.matrix prcomp
#' 
#' @export
Rtsne <- function (X, ...) {
  UseMethod("Rtsne", X)
}

#' @describeIn Rtsne Default Interface
#' @export
Rtsne.default <- function(X, dims=2, initial_dims=50, perplexity=30, theta=0.5, check_duplicates=TRUE, pca=TRUE,max_iter=1000,verbose=FALSE, is_distance=FALSE, Y_init=NULL, ...) {
  
  if (!is.logical(is_distance)) { stop("is_distance should be a logical variable")}
  if (!is.numeric(theta) || (theta<0.0) || (theta>1.0) ) { stop("Incorrect theta.")}
  if (nrow(X) - 1 < 3 * perplexity) { stop("Perplexity is too large.")}
  if (!is.matrix(X)) { stop("Input X is not a matrix")}
  if (!(max_iter>0)) { stop("Incorrect number of iterations.")}
  if (is_distance & !(is.matrix(X) & (nrow(X)==ncol(X)))) { stop("Input is not an accepted distance matrix") }
  if (!is.null(Y_init) & (nrow(X)!=nrow(Y_init) || ncol(Y_init)!=dims)) { stop("Incorrect format for Y_init.") }
  
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
  
  if (is.null(Y_init)) {
    init <- FALSE
    Y_init <- matrix()
  } else {
    init <- TRUE
  }
  
  Rtsne_cpp(X, dims, perplexity, theta,verbose, max_iter, is_distance, Y_init, init)
}

#' @describeIn Rtsne tsne on given dist object
#' @export
Rtsne.dist <- function(X,...,is_distance=TRUE) {
  X <- as.matrix(X)
  Rtsne(X, ..., is_distance=is_distance)
}

#' @describeIn Rtsne tsne on data.frame
#' @export
Rtsne.data.frame <- function(X,...) {
  X <- model.matrix(~.-1,X)
  Rtsne(X, ...)
}
