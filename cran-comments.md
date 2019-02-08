This is a rather large update of the package. Among other things, it now optionally uses OpenMP. I have attempted to follow the suggestions in the manual to ensure portability.

Another change is the license information: it was pointed out by a user that the licensing was not clear enough, since some files that are included are from a different project that uses an old-style 4-clause BSD license, while the rest of the code uses a regular 3-clause license. I have followed the practice used in the Rttf2pt1 package to update the LICENSE file. Please advise if this should be done differently.

## Test environments
* local OS X install
* win-builder (devel and release)
* ubuntu linux (devel, release and oldrelease) on Travis CI

## R CMD check results
1 NOTE: Possibly mis-spelled words in DESCRIPTION
These are not mis-spelled.
