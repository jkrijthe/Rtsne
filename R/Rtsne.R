#' Barnes-Hut implementation of t-Distributed Stochastic Neighbor Embedding 
#' 
#' Wrapper for the C++ implementation of Barnes-Hut t-Distributed Stochastic Neighbor Embedding. t-SNE is a method for constructing a low dimensional embedding of high-dimensional data, distances or similarities. Exact t-SNE can be computed by setting theta=0.0. 
#' 
#' Given a distance matrix \eqn{D} between input objects (which by default, is the euclidean distances between two objects), we calculate a similarity score in the original space p_ij. \deqn{ p_{j | i} = \frac{\exp(-\|D_{ij}\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|D_{ij}\|^2 / 2 \sigma_i^2)} } which is then symmetrized using: \deqn{ p_{i j}=\frac{p_{j|i} + p_{i|j}}{2n}.} The \eqn{\sigma} for each object is chosen in such a way that the perplexity of p_{j|i} has a value that is close to the user defined perplexity. This value effectively controls how many nearest neighbours are taken into account when constructing the embedding in the low-dimensional space.
#' For the low-dimensional space we use the Cauchy distribution (t-distribution with one degree of freedom) as the distribution of the distances to neighbouring objects:
#' \deqn{ q_{i j} = \frac{(1+ \| y_i-y_j\|^2)^{-1}}{\sum_{k \neq l} 1+ \| y_k-y_l\|^2)^{-1}}.} 
#' By changing the location of the objects y in the embedding to minimize the Kullback-Leibler divergence between these two distributions \eqn{ q_{i j}} and \eqn{ p_{i j}}, we create a map that focusses on small-scale structure, due to the asymmetry of the KL-divergence. The t-distribution is chosen to avoid the crowding problem: in the original high dimensional space, there are potentially many equidistant objects with moderate distance from a particular object, more than can be accounted for in the low dimensional representation. The t-distribution makes sure that these objects are more spread out in the new representation.
#' 
#' For larger datasets, a problem with the a simple gradient descent to minimize the Kullback-Leibler divergence is the computational complexity of each gradient step (which is \eqn{O(n^2)}). The Barnes-Hut implementation of the algorithm attempts to mitigate this problem using two tricks: (1) approximating small similarities by 0 in the \eqn{p_{ij}} distribution, where the non-zero entries are computed by finding 3*perplexity nearest neighbours using an efficient tree search. (2) Using the Barnes-Hut algorithm in the computation of the gradient which approximates large distance similarities using a quadtree. This approximation is controlled by the \code{theta} parameter, with smaller values leading to more exact approximations. When \code{theta=0.0}, the implementation uses a standard t-SNE implementation. The Barnes-Hut approximation leads to a \eqn{O(n log(n))} computational complexity for each iteration.
#' 
#' During the minimization of the KL-divergence, the implementation uses a trick known as early exaggeration, which multiplies the \eqn{p_{ij}}'s by 12 during the first 250 iterations. This leads to tighter clustering and more distance between clusters of objects. This early exaggeration is not used when the user gives an initialization of the objects in the embedding by setting \code{Y_init}. During the early exaggeration phase, a momentum term of 0.5 is used while this is changed to 0.8 after the first 250 iterations. All these default parameters can be changed by the user.
#' 
#' After checking the correctness of the input, the \code{Rtsne} function (optionally) does an initial reduction of the feature space using \code{\link{prcomp}}, before calling the C++ TSNE implementation. Since R's random number generator is used, use \code{\link{set.seed}} before the function call to get reproducible results.
#' 
#' If \code{X} is a data.frame, it is transformed into a matrix using \code{\link{model.matrix}}. If \code{X} is a \code{\link{dist}} object, it is currently first expanded into a full distance matrix.
#' 
#' @param X matrix; Data matrix (each row is an observation, each column is a variable)
#' @param index integer matrix; Each row contains the identity of the nearest neighbors for each observation 
#' @param distance numeric matrix; Each row contains the distance to the nearest neighbors in \code{index} for each observation
#' @param dims integer; Output dimensionality (default: 2)
#' @param initial_dims integer; the number of dimensions that should be retained in the initial PCA step (default: 50)
#' @param perplexity numeric; Perplexity parameter (should not be bigger than 3 * perplexity < nrow(X) - 1, see details for interpretation)
#' @param theta numeric; Speed/accuracy trade-off (increase for less accuracy), set to 0.0 for exact TSNE (default: 0.5)
#' @param check_duplicates logical; Checks whether duplicates are present. It is best to make sure there are no duplicates present and set this option to FALSE, especially for large datasets (default: TRUE)
#' @param pca logical; Whether an initial PCA step should be performed (default: TRUE)
#' @param partial_pca logical; Whether truncated PCA should be used to calculate principal components (requires the irlba package). This is faster for large input matrices (default: FALSE)
#' @param max_iter integer; Number of iterations (default: 1000)
#' @param verbose logical; Whether progress updates should be printed (default: global "verbose" option, or FALSE if that is not set)
#' @param ... Other arguments that can be passed to Rtsne
#' @param is_distance logical; Indicate whether X is a distance matrix (default: FALSE)
#' @param Y_init matrix; Initial locations of the objects. If NULL, random initialization will be used (default: NULL). Note that when using this, the initial stage with exaggerated perplexity values and a larger momentum term will be skipped.
#' @param pca_center logical; Should data be centered before pca is applied? (default: TRUE)
#' @param pca_scale logical; Should data be scaled before pca is applied? (default: FALSE)
#' @param normalize logical; Should data be normalized internally prior to distance calculations with \code{\link{normalize_input}}? (default: TRUE)
#' @param stop_lying_iter integer; Iteration after which the perplexities are no longer exaggerated (default: 250, except when Y_init is used, then 0)
#' @param mom_switch_iter integer; Iteration after which the final momentum is used (default: 250, except when Y_init is used, then 0) 
#' @param momentum numeric; Momentum used in the first part of the optimization (default: 0.5)
#' @param final_momentum numeric; Momentum used in the final part of the optimization (default: 0.8)
#' @param eta numeric; Learning rate (default: 200.0)
#' @param exaggeration_factor numeric; Exaggeration factor used to multiply the P matrix in the first part of the optimization (default: 12.0)
#' @param num_threads integer; Number of threads to use when using OpenMP, default is 1. Setting to 0 corresponds to detecting and using all available cores
#' 
#' @return List with the following elements:
#' \item{Y}{Matrix containing the new representations for the objects}
#' \item{N}{Number of objects}
#' \item{origD}{Original Dimensionality before TSNE (only when \code{X} is a data matrix)}
#' \item{perplexity}{See above}
#' \item{theta}{See above}
#' \item{costs}{The cost for every object after the final iteration}
#' \item{itercosts}{The total costs (KL-divergence) for all objects in every 50th + the last iteration}
#' \item{stop_lying_iter}{Iteration after which the perplexities are no longer exaggerated}
#' \item{mom_switch_iter}{Iteration after which the final momentum is used}
#' \item{momentum}{Momentum used in the first part of the optimization}
#' \item{final_momentum}{Momentum used in the final part of the optimization}
#' \item{eta}{Learning rate}
#' \item{exaggeration_factor}{Exaggeration factor used to multiply the P matrix in the first part of the optimization}
#' 
#' @section Supplying precomputed distances:
#' If a distance matrix is already available, this can be directly supplied to \code{Rtsne} by setting \code{is_distance=TRUE}.
#' This improves efficiency by avoiding recalculation of distances, but requires some work to get the same results as running default \code{Rtsne} on a data matrix.
#' Specifically, Euclidean distances should be computed from a normalized data matrix - see \code{\link{normalize_input}} for details.
#' PCA arguments will also be ignored if \code{is_distance=TRUE}.
#' 
#' NN search results can be directly supplied to \code{Rtsne_neighbors} to avoid repeating the (possibly time-consuming) search.
#' To achieve the same results as \code{Rtsne} on the data matrix, the search should be conducted on the normalized data matrix.
#' The number of nearest neighbors should also be equal to three-fold the \code{perplexity}, rounded down to the nearest integer.
#' Note that pre-supplied NN results cannot be used when \code{theta=0} as they are only relevant for the approximate algorithm.
#' 
#' Any kind of distance metric can be used as input.
#' In contrast, running \code{Rtsne} on a data matrix will always use Euclidean distances.
#'
#' @references Maaten, L. Van Der, 2014. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research, 15, p.3221-3245.
#' @references van der Maaten, L.J.P. & Hinton, G.E., 2008. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research, 9, pp.2579-2605.
#' 
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' iris_matrix <- as.matrix(iris_unique[,1:4])
#' 
#' # Set a seed if you want reproducible results
#' set.seed(42)
#' tsne_out <- Rtsne(iris_matrix,pca=FALSE,perplexity=30,theta=0.0) # Run TSNE
#' 
#' # Show the objects in the 2D tsne representation
#' plot(tsne_out$Y,col=iris_unique$Species, asp=1)
#' 
#' # data.frame as input
#' tsne_out <- Rtsne(iris_unique,pca=FALSE, theta=0.0)
#' 
#' # Using a dist object
#' set.seed(42)
#' tsne_out <- Rtsne(dist(normalize_input(iris_matrix)), theta=0.0)
#' plot(tsne_out$Y,col=iris_unique$Species, asp=1)
#' 
#' set.seed(42)
#' tsne_out <- Rtsne(as.matrix(dist(normalize_input(iris_matrix))),theta=0.0)
#' plot(tsne_out$Y,col=iris_unique$Species, asp=1)
#' 
#' # Supplying starting positions (example: continue from earlier embedding)
#' set.seed(42)
#' tsne_part1 <- Rtsne(iris_unique[,1:4], theta=0.0, pca=FALSE, max_iter=350)
#' tsne_part2 <- Rtsne(iris_unique[,1:4], theta=0.0, pca=FALSE, max_iter=650, Y_init=tsne_part1$Y)
#' plot(tsne_part2$Y,col=iris_unique$Species, asp=1)
#' \dontrun{
#' # Fast PCA and multicore
#' 
#' tsne_out <- Rtsne(iris_matrix, theta=0.1, partial_pca = TRUE, initial_dims=3)
#' tsne_out <- Rtsne(iris_matrix, theta=0.1, num_threads = 2)
#' }
#' @useDynLib Rtsne, .registration = TRUE
#' @import Rcpp
#' @importFrom stats model.matrix na.fail prcomp
#' 
#' @export
Rtsne <- function (X, ...) {
  UseMethod("Rtsne", X)
}

#' @describeIn Rtsne Default Interface
#' @export
Rtsne.default <- function(X, dims=2, initial_dims=50, 
                          perplexity=30, theta=0.5, 
                          check_duplicates=TRUE, 
                          pca=TRUE, partial_pca=FALSE, max_iter=1000,verbose=getOption("verbose", FALSE), 
                          is_distance=FALSE, Y_init=NULL, 
                          pca_center=TRUE, pca_scale=FALSE, normalize=TRUE,
                          stop_lying_iter=ifelse(is.null(Y_init),250L,0L), 
                          mom_switch_iter=ifelse(is.null(Y_init),250L,0L), 
                          momentum=0.5, final_momentum=0.8,
                          eta=200.0, exaggeration_factor=12.0, num_threads=1, ...) {
  
  if (!is.logical(is_distance)) { stop("is_distance should be a logical variable")}
  if (!is.matrix(X)) { stop("Input X is not a matrix")}
  if (is_distance & !(is.matrix(X) & (nrow(X)==ncol(X)))) { stop("Input is not an accepted distance matrix") }
  if (!(is.logical(pca_center) && is.logical(pca_scale)) ) { stop("pca_center and pca_scale should be TRUE or FALSE")}
  if (!is.wholenumber(initial_dims) || initial_dims<=0) { stop("Incorrect initial dimensionality.")}
  if (!is.wholenumber(num_threads) || num_threads<0) { stop("Incorrect number of threads.")}
  tsne.args <- .check_tsne_params(nrow(X), dims=dims, perplexity=perplexity, theta=theta, max_iter=max_iter, verbose=verbose, 
        Y_init=Y_init, stop_lying_iter=stop_lying_iter, mom_switch_iter=mom_switch_iter, 
        momentum=momentum, final_momentum=final_momentum, eta=eta, exaggeration_factor=exaggeration_factor)
 
  # Check for missing values
  X <- na.fail(X)
  
  # Apply PCA
  if (!is_distance) { 
    if (pca) {
      if(verbose) cat("Performing PCA\n")
      if(partial_pca){
        if (!requireNamespace("irlba", quietly = TRUE)) {stop("Package \"irlba\" is required for partial PCA. Please install it.", call. = FALSE)}
        X <- irlba::prcomp_irlba(X, n = initial_dims, center = pca_center, scale = pca_scale)$x
      }else{
        if(verbose & min(dim(X))>2500) cat("Consider setting partial_pca=TRUE for large matrices\n")
        X <- prcomp(X, retx=TRUE, center = pca_center, scale. = pca_scale, rank. = initial_dims)$x
      }
    }
    if (check_duplicates) {
      if (any(duplicated(X))) { stop("Remove duplicates before running TSNE.") }
    }
    if (normalize) {
      X <- normalize_input(X)
    }
    X <- t(X) # transposing for rapid column-major access.
  } else {
    # Compute Squared distance if we are using exact TSNE
    if (theta==0.0) {
      X <- X^2
    }
  }
 
  out <- do.call(Rtsne_cpp, c(list(X=X, distance_precomputed=is_distance, num_threads=num_threads), tsne.args))
  out$Y <- t(out$Y) # Transposing back.
  info <- list(N=ncol(X))
  if (!is_distance) { out$origD <- nrow(X) } # 'origD' is unknown for distance matrices.
  out <- c(info, out, .clear_unwanted_params(tsne.args))
  class(out) <- c("Rtsne","list")
  out
}

#' @describeIn Rtsne tsne on given dist object
#' @export
Rtsne.dist <- function(X,...,is_distance=TRUE) {
  X <- as.matrix(na.fail(X))
  Rtsne(X, ..., is_distance=is_distance)
}

#' @describeIn Rtsne tsne on data.frame
#' @export
Rtsne.data.frame <- function(X,...) {
  X <- model.matrix(~.-1,na.fail(X))
  Rtsne(X, ...)
}
