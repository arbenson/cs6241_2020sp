module cp
export ⊙, cp_als, matricize
"""
Alternating least square for CP.
Please refer to Austin's lecture notes and Kolda & Bader.
"""

"""
    matricize(ts::AbstractArray{Float64}, n::Int64)

Matricize a tensor ts in mode n. See Kolda & Bader for the formula.
"""
function matricize(ts::AbstractArray{Float64}, n::Int64)
    J = [reduce(*, [size(ts, m) for m = 1:k-1 if m != n], init=1) for k = 1:ndims(ts)]
    mx = zeros(size(ts, n), prod([size(ts, m) for m = 1:ndims(ts) if m != n]))
    for i in CartesianIndices(ts)
        j = 1 + sum([(i[k] - 1) * J[k] for k = 1:ndims(ts) if k != n])
        mx[i[n], j] = ts[i] end
    return mx end

"""
    ⊙(x::AbstractMatrix{Float64}, y::AbstractMatrix{Float64})

Compute the Khatri-Rao product
"""
⊙(x::AbstractMatrix{Float64}, y::AbstractMatrix{Float64}) = hcat([kron(a, b) for (a, b) in zip(eachcol(x), eachcol(y))]...)

"""
    cp_als(ts, r::Int64, n_iter::Int64)

Compute the CP using alternating least-square for general tensors.

# Arguments
- `ts`: The tensor to decompose.
- `r`: The rank of the decomposition.
- `n_iter`: the number of aternating least-square iteration.
"""
function cp_als(ts, r::Int64, n_iter::Int64)
    ls_step(a, mxs) =
        let v = reduce((a, b) -> a .* b, [(m' * m) for m in mxs]),
            k = reduce(⊙, mxs)
            (v \ (k' * a'))' end

    factors = [rand(i, r) for i in size(ts)]
    for _ = 1:n_iter
        for n = 1:ndims(ts)
            factors[n] = let mxs = reverse(vcat(factors[1:n-1], factors[n+1:end]))
                ls_step(matricize(ts, n), mxs) end end end
    return factors end

end  # module cp
