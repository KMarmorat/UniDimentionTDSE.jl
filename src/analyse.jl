export  P_windows
function P_windows(ψ,H_0::SymTridiagonal,E::Real,γ::Real)
    """Compute the probability P(E,n,γ) at E"""
    H1 = H_0 + I*(-E +√(im) *γ )
    H2 = H_0 + I*(-E -√(im) *γ )

    norm(H2\(H1\ψ))^2
end

function wigner(ψ, x::StepRangeLen,i::Integer,p::Real)
    @assert(length(ψ) == length(x))
    n = length(x)

    integrate  = j -> conj(ψ[i+j])*ψ[i-j]*exp(-2im*p*x[j+x.offset])*x.step
    minimum = max(1-i,i-n)
    maximum = min(i-1,-i+n)
    (maximum == 0) && return 0

    convert(ComplexF64,sum((integrate(j) for j in minimum:maximum)))

end

export plot_wigner
function plot_wigner(ψ,x::StepRangeLen;Δx::Float64=0.)
    "should be use like surface(plot_wigner()...)"
    xs = iszero(Δx) ? x : range(x[1],x[end];step=Δx)
    ps = xs
    steps = iszero(Δx) ? 1 : Int(fld(Δx,step(x)))
    wigners = [wigner(ψ,x,i,p) for i in 1:steps:length(x), p in ps]
    return (xs,ps,abs2.(wigners))
end
