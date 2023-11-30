export  P_windows, Sum_P_windows
function plot_P_windows(ψ,V,x::StepRangeLen,E,γ;padding::Integer=1)
    @assert(length(x) == length(ψ))

    x_tilde = padding*x[1]:Float64(x.step):padding*x[end]
    ψ_tilde = zeros(eltype(ψ),length(x_tilde))
    N = (length(ψ)-1) ÷ 2
    ψ_tilde[(padding-1)*N+1:(padding+1)*(N) + 1] .= ψ
    H = Hamiltonian(V,x_tilde)
    return [P_windows(ψ_tilde,H,ϵ,γ) for ϵ in E]
end

function P_windows(ψ,H_0::SymTridiagonal,E::Real,γ::Real)
    """Compute the probability P(E,n,γ) at E"""
    H1 = H_0 + I*(-E +√(im) *γ)
    H2 = H_0 + I*(-E -√(im) *γ)
    normalize(ψ)
    ξ = (H1\(H2\ψ)*γ^2)
    norm(ξ)^2
end

function Sum_P_windows(ψ,V,x,t,RangeE,Elim::Real;γ::Real=0.02,padding::Integer=1,μ::Integer=1)
    x_tilde = x[1]:Float64(x.step):padding*x[end]

    ψ_tilde = zeros(eltype(ψ),length(x_tilde))

    N = (length(ψ))
    ψ_tilde[1:N] .= ψ

    H = Hamiltonian(x_tilde,V;μ)

    sum(E -> P_windows(ψ_tilde,H,E,γ)*(E>Elim ? 1/(2γ) : 1)*exp(im*E*t),RangeE)
end

function wigner(ψ, x::StepRangeLen,i::Integer,p::Real)
    @assert(length(ψ) == length(x))
    n = length(x)

    integrate  = j -> conj(ψ[i+j])*ψ[i-j]*exp(-2im*p*j*x.step)*x.step
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
