module UniDimentionTDSE
using LinearAlgebra,Statistics,DelimitedFiles
export Hamiltonian,simulate

struct SimulationParameter
    Δx::Float64
    a::Float64
    Δt::Float64
    time::Float64
    Nt::Int64
    Neig::Int64
    Filename::String
end

function writeToFile(ψ,param::SimulationParameter,eigVecs,F,t)
    open(param.Filename,"a") do io
        pop = reshape([abs(sum(ψ.*ϕ)) for ϕ in eachcol(eigVecs)],1,param.Neig);
        writedlm(io,[t  pop F(t) angle(ψ[end÷2])])
    end;
end

function initiateFile(param,syntax::String)
    open(param.Filename,"w") do io
        write(io,"#Syntax is:" * syntax)
        writedlm(io,[param.Δx param.a param.Δt param.time param.Neig])
    end;
end

function readFile(param)
    open(param.Filename,"r") do io
        R = readdlm(io,)
        (param.Δx,param.a,param.Δt,param.time,param.Neig) = R[1,1:4]
        return [R[:,x] for x in 2:(2+param.Neig)]
    end;
end

struct Gaussian{T}
   x::T 
   p::T
   α::Complex{T}
   γ::Complex{T}
end

function gaussian(x::AbstractRange,g::Gaussian)
    @. exp(-g.α * (x-g.x)^2 + im*g.p*(x-g.x) + im*g.γ)
end

function propagate!(ψ,Htop,Hbottom)
    "Update the wave function"
    ψ .= Hbottom\(Htop*ψ)
end

function buildCrankNicolson!(H,Htop,Hbottom,Δt)
    "Update the upper and lower part of Crank Nicolson"
    Htop.dv .= 1 .- im * H.dv * Δt/2
    Hbottom.dv .= 1 .+ im * H.dv * Δt/2
end

function buildCrankNicolson(H,Δt)
    "Return the upper and lower part of Crank Nicolson"
    Htop = I - im * H * Δt/2
    Hbottom = I + im * H * Δt/2
    (Htop,Hbottom)
end

function Hamiltonian!(H::SymTridiagonal,Hdiag_0,x::AbstractRange,F,t)
    "Update the Hamiltonian"
    @. H.dv = Hdiag_0 + x*F(t)
end

function Hamiltonian(x::AbstractRange,V)
    "Create the time independant Hamiltonian"
    Δx = step(x)
    midline = V.(x)   .+ 1/(Δx)^2
    topline = zero(x) .- 1/(2*(Δx)^2)
    SymTridiagonal((midline),(topline))
end

function simulate(ψ,param::SimulationParameter,x,V)
    simulate(ψ,param,x,V,(t)->0)
end

function simulate(ψ,param::SimulationParameter,x,V,F)
    @assert (iszero(imag(param.Δt)) || iszero(real(param.Δt)==0))

    H = Hamiltonian(x,V)

    _,eigVecs = eigen(H,1:param.Neig)

    
    H_0 = copy(H)
    (Htop,Hbottom) = buildCrankNicolson(H,param.Δt)

    for i = 1:param.Nt
        t = i*param.Δt

        Hamiltonian!(H,H_0.dv,x,F,t)
        buildCrankNicolson!(H,Htop,Hbottom,param.Δt)
        propagate!(ψ,Htop,Hbottom)

        iszero(real(param.Δt)) && normalize!(ψ * param.Δx)
        writeToFile(ψ,param,eigVecs,F,t)
    end
end

function test()
    param = SimulationParameter(
    0.001,
    5,
    0.001,
    3.14,
    3140,
    3,
    "Test"
    )
    syntax = "\n"
    initiateFile(param,syntax::String)

    V = x -> 1/2*x.^2
    x = range(-5,5;step= 0.001)
    
    H =Hamiltonian(x,V)

    epsilon ,ψ_0 = eigen(H,1:1)
    normalize!(ψ_0 * param.Δx)

    ψ_0 = convert(Matrix{Complex},ψ_0)
    simulate(ψ_0,param,x,V)
end

end
