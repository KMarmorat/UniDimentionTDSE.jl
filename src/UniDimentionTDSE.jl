module UniDimentionTDSE
using LinearAlgebra,DelimitedFiles
export Hamiltonian,simulate,test,getEigen,test_wigner
include("absorbtion.jl")
include("analyse.jl")

struct SimulationParameter
    Δx::Float64
    a::Float64
    Δt::Float64
    time::Float64
    Nt::Int64
    Neig::Int64
    Filename::String
end

function writeToFile(ψ,param::SimulationParameter,eigVecs,F,t,io,extrafunctions...)

    pop = reshape([dot(ψ,normalize!(ϕ)*param.Δx) for ϕ in eachcol(eigVecs)],1,param.Neig)
    extra = reshape([f(ψ,t) for f in extrafunctions],1,length(extrafunctions))
    ψ_norm = norm(ψ)*param.Δx

    writedlm(io,[t  pop F(t) ψ_norm extra],';')

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

function Hamiltonian!(H::SymTridiagonal,Hdiag_0,x::AbstractRange,F,multF,t)
    "Update the Hamiltonian"
    @. H.dv = Hdiag_0 + F(t)*multF
end

function Hamiltonian(V,param::SimulationParameter)
    x = buildx(param)
    Hamiltonian(V,x)
end

function Hamiltonian(V,x::AbstractRange)
    "Create the time independant Hamiltonian"
    Δx = step(x)
    midline = V.(x)   .+ 1/(Δx)^2
    topline = zero(x) .- 1/(2*(Δx)^2)
    SymTridiagonal((midline),(topline))
end

function simulate(ψ,param::SimulationParameter,V)
    simulate(ψ,param,V,(t)->0)
end
function simulate(ψ,param::SimulationParameter,V,F::Function,extrafunctions...)
    simulate(ψ,param,V,F,buildx(param))
end

function simulate(ψ,param::SimulationParameter,V,F::Function,multF::Vector,extrafunctions...)
    @assert (iszero(imag(param.Δt)) || iszero(real(param.Δt)==0))
    x = buildx(param)

    H = Hamiltonian(V,x)
    H_0 = copy(H)

    _,eigVecs = getEigen(V,param;irange= 1:param.Neig)

    
    (Htop,Hbottom) = buildCrankNicolson(H,param.Δt)

    open(param.Filename,"w") do io
        for i = 1:param.Nt
            t = i*param.Δt

            Hamiltonian!(H,H_0.dv,x,F,multF,t)
            buildCrankNicolson!(H,Htop,Hbottom,param.Δt)
            propagate!(ψ,Htop,Hbottom)

            iszero(real(param.Δt)) && begin  normalize!(ψ); ψ/=param.Δx end

            writeToFile(ψ,param,eigVecs,F,t,io,extrafunctions...)
        end
    end

    open(param.Filename * ".wavefunctions","w") do io
        writedlm(io,ψ,';')
    end
end

function getEigen(V,param::SimulationParameter;irange::UnitRange=1:1)
    x = range(-param.a,param.a;step= param.Δx)
    getEigen(V,x;irange)
end
function getEigen(V,x::StepRangeLen;irange=1:1)

    H = Hamiltonian(V,x)
    E, ψs =  eigen(H,irange)

    for ψ in eachcol(ψs)
        normalize!(ψ)
        ψ ./= sqrt(step(x))
    end

    (E,ψs)
end
function buildx(param::SimulationParameter)
    range(-param.a,param.a;step=param.Δx)
end


function test()
    param = SimulationParameter(
    0.01,
    10,
    0.01,
    31.400,
    3140,
    20,
    "Test"
    )
    #syntax = "\n"
    #initiateFile(param,syntax::String)
    #rm(param.Filename)
    open(param.Filename,"w") do io
        write(io,"")
    end

    V = x -> 1/2*x.^2
    x = range(param)

    
    _ ,eig = getEigen(V,param ;irange = 1:3)
    ψ_0 = eig[:,1] 

    ψ_0 = convert(Array{Complex},ψ_0)

    @show norm(ψ_0)*param.Δx

    F(t) = 0.5*sin(t*0.8)*(t<=6π)

    simulate(ψ_0,param,V,F)
end
function test_wigner()
    param = SimulationParameter(
    0.01,
    5,
    0.1,
    31.400,
    3140,
    20,
    "Test"
    )
    V = x -> 1/2*x.^2
    x = buildx(param)

    Δx = 0.1
    _,ψ = getEigen(V,param)

    plot_wigner(ψ,x;Δx=Δx)
end
function test_P_windows()
    param = SimulationParameter(
    0.01,
    5,
    0.1,
    31.400,
    3140,
    20,
    "Test"
    )
    γ = 0.00006
    V = x -> 1/2*x.^2
    x = buildx(param)

    _,ψ = getEigen(V,param;irange=1:3)
    ψ = sum(ψ,dims=2)
    H = Hamiltonian(V,x)
    E = range(0.4,3;step=0.0001)
    Energy = [P_windows(ψ,H,ϵ,γ) for ϵ in E]

    (E,Energy)
    
end
function test_streaking()
    omega = 1.5
    N = 20
    Delta_t = 0.01
    param = SimulationParameter(
    0.0001,
    50,
    Delta_t,
    2N/omega*π,
    Int(floor(2N/omega*π / Delta_t)),
    20,
    "Test"
    )
    @show param
    a = 0.707
    V(x) = -1/sqrt(x^2 + a^2)

    x = buildx(param)
    E,ψ = getEigen(V,param)

    ψ = convert(Array{Complex},ψ)
    F(t) = 0.01*sin(omega/(2N)*t)^2*sin(omega*t)
    simulate(ψ,param,V,F)
    (V,param)
end
function hello()
    println("Hello")
end
end
