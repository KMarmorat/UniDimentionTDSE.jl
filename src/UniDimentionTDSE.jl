module UniDimentionTDSE
using LinearAlgebra,Statistics
export Hamiltonian,simulate,test

struct Gaussian{T}
   x::T 
   p::T
   α::Complex{T}
   γ::Complex{T}
end

function propagate!(ψ,Htop,Hbottom)
    ψ .= Hbottom\(Htop*ψ)
end

function buildCrankNicolson!(H,Htop,Hbottom,Δt)
    Htop.dv .= 1 .- im * H.dv * Δt/2
    Hbottom.dv .= 1 .+ im * H.dv * Δt/2
end

function buildCrankNicolson(H,Δt)
    Htop = I - im * H * Δt/2
    Hbottom = I + im * H * Δt/2
    (Htop,Hbottom)
end

function Hamiltonian!(H::SymTridiagonal,Hdiag_0,x::AbstractRange,F,t)
    H.dv .= Hdiag_0 .+ F(x,t)
end

function Hamiltonian(x::AbstractRange,V)
    Δx = step(x)
    midline = V.(x)   .+ 1/(Δx)^2
    topline = zero(x) .- 1/(2*(Δx)^2)
    SymTridiagonal(Complex.(midline),Complex.(topline))
end

function simulate(ψ,Nt::Integer,Δt::Number,V,x,f)
    simulate(ψ,Nt,Δt,V,(x,t)->0,x,f)
end

function simulate(ψ,Nt::Integer,Δt::Number,V,F,x,f)
    @assert (iszero(imag(Δt)) || iszero(real(Δt)==0))

    H = Hamiltonian(x,V)
    Hdiag_0 = copy(H.dv)
    (Htop,Hbottom) = buildCrankNicolson(H,Δt)

    Values = zeros(Float64,Nt)

    for i = 1:Nt
        t = i*Δt

        Hamiltonian!(H,Hdiag_0,x,F,t)
        buildCrankNicolson!(H,Htop,Hbottom,Δt)
        propagate!(ψ,Htop,Hbottom)

        iszero(real(Δt)) && normalize!(ψ)
        Values[i] = f(ψ) 
    end
    (ψ,Values)
end


function gaussian(x::AbstractRange,g::Gaussian)
    @. exp(-g.α * (x-g.x)^2 + im*g.p*(x-g.x) + im*g.γ)
end


function test_phase(T,Nt)
    Δt = T/Nt
    V = x -> 1/2*x.^2
    x = range(-5,5;step= 0.001)
    g = Gaussian{Float64}(0,0,1/2,0)

    ϕ = ψ -> angle(ψ[end÷2])


    ψ_1 = gaussian(x,g)
    ψ_0 = copy(ψ_1)

    normalize!(ψ_0)
    normalize!(ψ_1)


    (ψ_1*exp(-im*T/2),simulate(ψ_0,Nt,Δt,V,x,ϕ))
end

end
