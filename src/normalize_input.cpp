#include "Rcpp.h"
#include <vector>

// Function that performs the matrix normalization.
// [[Rcpp::export]]
Rcpp::NumericMatrix normalize_input_cpp(Rcpp::NumericMatrix input) {
    // Rows are observations, columns are variables.
    Rcpp::NumericMatrix output=Rcpp::clone(input);
    const int N=output.nrow(), D=output.ncol();

    // Running through each column and centering it.
    Rcpp::NumericMatrix::iterator oIt=output.begin();
    for (int d=0; d<D; ++d) {
        double cur_mean=0;

        Rcpp::NumericMatrix::iterator ocopy=oIt;
        for (int n=0; n<N; ++n, ++ocopy) {
            cur_mean += *ocopy;
        }
        cur_mean /= N;

        for (int n=0; n<N; ++n, ++oIt) {
            *oIt -= cur_mean;
        }
    }

    // Computing the maximum deviation and scaling all elements.
    double max_X = .0;
    for (oIt=output.begin(); oIt!=output.end(); ++oIt) {
        const double tmp = fabs(*oIt);
        if(tmp > max_X) max_X = tmp;
    }
    for (oIt=output.begin(); oIt!=output.end(); ++oIt) {
        *oIt /= max_X;
    }
    return output;
}
