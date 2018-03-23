using LinAlgExtensions
using Test
using LinearAlgebra
using SparseArrays
using SuiteSparse

@testset "utils"     begin include("utils.jl") end
@testset "randutils" begin include("randutils.jl") end
@testset "pinvfact"  begin include("pinv.jl") end

