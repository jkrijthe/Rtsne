#' Normalize input data matrix
#'
#' Mean centers each column of an input data matrix so that it has a mean of zero.
#' Scales the entire matrix so that the largest absolute of the centered matrix is equal to unity.
#'
#' @param X matrix; Input data matrix with rows as observations and columns as variables/dimensions.
#'
#' @details
#' Normalization avoids numerical problems when the coordinates (and thus the distances between observations) are very large.
#' Directly computing distances on this scale may lead to underflow when computing the probabilities in the t-SNE algorithm.
#' Rescaling the input values mitigates these problems to some extent.
#' 
#' @author
#' Aaron Lun
#'
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' iris_matrix <- as.matrix(iris_unique[,1:4])
#' X <- normalize_input(iris_matrix)
#' colMeans(X)
#' range(X)
#' @export
normalize_input <- function(X) {
    # Using the original C++ code from bhtsne to do mean centering, even though it would be simple to do with sweep().
    # This is because R's sums are more numerically precise, so for consistency with the original code, we need to use the naive C++ version.
    normalize_input_cpp(X)
}
