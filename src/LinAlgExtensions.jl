module LinAlgExtensions

if VERSION >= v"0.7.0-DEV"
    using LinearAlgebra
    using SparseArrays
    using SuiteSparse
end

include("utils.jl")
include("randutils.jl")
include("pinv.jl")

end # module
