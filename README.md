# LinAlgExtensions

[![Build Status](https://travis-ci.org/KlausC/LinAlgExtensions.jl.svg?branch=master)](https://travis-ci.org/KlausC/LinAlgExtensions.jl)
[![Coverage Status](https://coveralls.io/repos/KlausC/LinAlgExtensions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/KlausC/LinAlgExtensions.jl?branch=master)
[![codecov.io](http://codecov.io/github/KlausC/LinAlgExtensions.jl/coverage.svg?branch=master)](http://codecov.io/github/KlausC/LinAlgExtensions.jl?branch=master)

## Consistent interface for QR-Factorization

## Pseudo-Inverse Factorization

For sparse matrices `A`, there is currently algorithm implemented to calculate
`pinv(A) * rhs` and `pinv(A)' * rhs` for arbitrary right hand sides `rhs`.

An algorithm is provided, which uses two QR-Factorizations.
Without considering the column- and row permutation, which complicate the description
it is as follows:

    If `A = Q * R` where `R` has maximal rank (identical to the rank of `A`)

    and `R' = q * r`

We have `A = Q * r' * q'` and `pinv(A) = q * inv(r)' * Q'`.

### usage:
```
    pin = pinvfact(A)

    x = pin \ rhs
```
## Random test matrices

randorth: orthogonal columns or rows

randrealsv: arbitrary shape matrices with gibven singular values

randsparse: sparse random matrix with given shape and rank

