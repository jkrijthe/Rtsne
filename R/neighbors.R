#' @rdname Rtsne
#' @export
Rtsne_neighbors <- function(index, distance, dims=2, perplexity=30, theta=0.5,
        max_iter=1000,verbose=getOption("verbose", FALSE),
        Y_init=NULL,
        stop_lying_iter=ifelse(is.null(Y_init),250L,0L),
        mom_switch_iter=ifelse(is.null(Y_init),250L,0L),
        momentum=0.5, final_momentum=0.8,
        eta=200.0, exaggeration_factor=12.0, num_threads=1, ...) {

    if (!is.matrix(index)) { stop("Input index is not a matrix") }
    if (!identical(dim(index), dim(distance))) { stop("index and distance matrices should have the same dimensions") }
    R <- range(index)
    if (any(R < 1 | R > nrow(index) | !is.finite(R))) { stop("invalid indices") }
    tsne.args <- .check_tsne_params(nrow(index), dims=dims, perplexity=perplexity, theta=theta, max_iter=max_iter, verbose=verbose,
            Y_init=Y_init, stop_lying_iter=stop_lying_iter, mom_switch_iter=mom_switch_iter,
            momentum=momentum, final_momentum=final_momentum, eta=eta, exaggeration_factor=exaggeration_factor)

    # Transposing is necessary for fast column-major access to each sample, -1 for zero-indexing.
    out <- do.call(Rtsne_nn_cpp, c(list(nn_dex=t(index - 1L), nn_dist=t(distance), num_threads=num_threads), tsne.args))
    out$Y <- t(out$Y) # Transposing back.
    c(list(N=nrow(index)), out, .clear_unwanted_params(tsne.args))
}
