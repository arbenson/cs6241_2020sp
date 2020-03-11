using Test
using LinearAlgebra
push!(LOAD_PATH, pwd())

using cp

"""
    ⊗(u, v::AbstractVector{Float64})

Compute the tensor product of a tensor product (can be a vector) and a vector.
This is different from the kronecker product.
"""
⊗(u, v::AbstractVector{Float64}) =
    reshape(kron(v, reshape(u, prod(size(u)))), (size(u)..., size(v)...))

"""
    tensor_from_matrices(mxs::Vararg{AbstractMatrix{Float64}})

Compute the tensor X = [[A, B, C, ...]] = ∑a_r ⊗ b_r ⊗ c_r ⊗ ....,
where a_r are the columns of A.
"""

tensor_from_matrices(mxs::Vararg{AbstractMatrix{Float64}}) =
    sum([reduce(⊗, cols) for cols in zip([eachcol(mx) for mx in mxs]...)])


test_khatri_rao_product() = let A = [1.0 2.0; 2.0 1.0],
                                B = [1.0 2.0; 3.0 4.0],
                                expect = [1.0 4.0; 3.0 8.0; 2.0 2.0; 6.0 4.0]
    @test A ⊙ B == expect end

test_tensor_product() = let v = [1.0, 2.0, 3.0]
    @test norm(v ⊗ reverse(v) - v .* reverse(v)') < 1e-7 end

test_tensor_product_2() = let u = [2.0, 1.0], v = [1.0, 2.0, 3.0]
    @test norm(v ⊗ u - v .* u') < 1e-7 end

test_tensor_product_order_3() = let u = [1.0, 2.0, 4.0],
                                    v = [3.0, 1.0, 3.0],
                                    w = [2.0, 1.0, 3.0]
    @test norm(u ⊗ v ⊗ w - reshape(kron(w, v, u), (3, 3, 3))) < 1e-7 end

test_tensor_from_matrices() = let U = [1.0 2.0;
                                       2.0 1.0],
                                  V = [1.0 2.0;
                                       3.0 4.0],
                                  W = [1.0 3.0;
                                       2.0 4.0]
    @test norm(tensor_from_matrices(U, V, W) - reshape([13.0 27.0 18.0 38.0;
                                                        8.0 18.0 12.0 28.0],
                                                       (2, 2, 2))) < 1e-7 end

test_matricize() = let x = reshape([1.0 4.0 7.0 10.0 13.0 16.0 19.0 22.0;
                                    2.0 5.0 8.0 11.0 14.0 17.0 20.0 23.0;
                                    3.0 6.0 9.0 12.0 15.0 18.0 21.0 24.0], (3, 4, 2)),
    expect_1 = [1.0 4.0 7.0 10.0 13.0 16.0 19.0 22.0;
                2.0 5.0 8.0 11.0 14.0 17.0 20.0 23.0;
                3.0 6.0 9.0 12.0 15.0 18.0 21.0 24.0],
    expect_2 = [1.0   2.0   3.0   13.0  14.0  15.0;
                4.0   5.0   6.0   16.0  17.0  18.0;
                7.0   8.0   9.0   19.0  20.0  21.0;
                10.0  11.0  12.0  22.0  23.0  24.0],
    expect_3 = [1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0   10.0  11.0  12.0
                13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0  24.0]
    @test matricize(x, 1) == expect_1
    @test matricize(x, 2) == expect_2
    @test matricize(x, 3) == expect_3 end

test_cp_als_cube() = let u = [1.0, 2.0, 4.0], v = [3.0, 1.0, 3.0], w = [2.0, 1.0, 3.0]
    ts = u ⊗ v ⊗ w
    x, y, z = cp_als(ts, 1, 50)
    @test norm(tensor_from_matrices(x, y, z) - ts) < 1e-7 end

test_cp_als_rectangle() = let u = [1.0, 2.0], v = [3.0, 1.0, 3.0], w = [2.0, 1.0, 3.0, 4.0]
    ts = u ⊗ v ⊗ w
    x, y, z = cp_als(ts, 1, 200)
    @test norm(tensor_from_matrices(x, y, z) - ts) < 1e-7 end

test_cp_als_rank_3() = let U = [1.0 2.0 4.0;
                                5.0 4.0 2.0;
                                7.0 2.0 2.0],
                           V = [3.0 1.0 3.0;
                                3.0 8.0 1.0;
                                4.0 3.0 7.0],
                           W = [2.0 7.0 3.0;
                                4.0 3.0 9.0;
                                1.0 2.0 9.0]
    ts = tensor_from_matrices(U, V, W)
    x, y, z = cp_als(ts, 3, 1000)
    @test norm(tensor_from_matrices(x, y, z) - ts) < 1e-7 end

test_cp_als_rank_4() = let U = [1.0 2.0 4.0 2.0;
                                5.0 4.0 2.0 5.0;
                                7.0 2.0 2.0 3.0;
                                7.0 2.0 2.0 9.0],
                           V = [3.0 1.0 3.0 1.0;
                                3.0 8.0 1.0 3.0;
                                3.0 6.0 4.0 9.0;
                                4.0 3.0 7.0 6.0],
                           W = [2.0 7.0 9.0 3.0;
                                4.0 3.0 2.0 9.0;
                                2.0 8.0 8.0 1.0;
                                1.0 2.0 3.0 9.0]
    ts = tensor_from_matrices(U, V, W)
    x, y, z = cp_als(ts, 4, 8000)
    @test norm(tensor_from_matrices(x, y, z) - ts) < 1e-7 end

test_khatri_rao_product()
test_tensor_product()
test_tensor_product_2()
test_tensor_product_order_3()
test_matricize()
test_tensor_from_matrices()
test_cp_als_cube()
test_cp_als_rectangle()
test_cp_als_rank_3()
test_cp_als_rank_4()

