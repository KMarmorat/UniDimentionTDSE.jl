module UniDimentionTDSE
using LinearAlgebra,DelimitedFiles
export Hamiltonian,simulate,test,getEigen

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

    pop = reshape([abs2(dot(ψ,normalize!(ϕ)/param.Δx)) for ϕ in eachcol(eigVecs)],1,param.Neig)
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

function Hamiltonian!(H::SymTridiagonal,Hdiag_0,x::AbstractRange,F,t)
    "Update the Hamiltonian"
    @. H.dv = Hdiag_0 + x*F(t)
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

function simulate(ψ,param::SimulationParameter,V,F,extrafunctions...)
    @assert (iszero(imag(param.Δt)) || iszero(real(param.Δt)==0))
    x = range(-param.a,param.a;step= param.Δx)

    H = Hamiltonian(V,x)
    H_0 = copy(H)

    _,eigVecs = eigen(H,1:param.Neig)

    
    (Htop,Hbottom) = buildCrankNicolson(H,param.Δt)

    open(param.Filename,"a") do io
        for i = 1:param.Nt
            t = i*param.Δt

            Hamiltonian!(H,H_0.dv,x,F,t)
            buildCrankNicolson!(H,Htop,Hbottom,param.Δt)
            propagate!(ψ,Htop,Hbottom)

            iszero(real(param.Δt)) && begin  normalize!(ψ); ψ/=param.Δx end

            writeToFile(ψ,param,eigVecs,F,t,io,extrafunctions...)
        end
    end
end

function getEigen(V,param::SimulationParameter;irange::UnitRange=1:1)
    x = range(-param.a,param.a;step= param.Δx)
    H = Hamiltonian(V,x)
    E, ψs =  eigen(H,irange)

    for ψ in eachcol(ψs)
        normalize!(ψ)
        ψ ./= param.Δx
    end

    (E,ψs)
end


function test()
    param = SimulationParameter(
    0.01,
    5,
    0.01,
    31.400,
    3140,
    3,
    "Test"
    )
    #syntax = "\n"
    #initiateFile(param,syntax::String)
    #rm(param.Filename)

    V = x -> 1/2*x.^2
    x = range(-5,5;step= 0.01)

    
    _ ,eig = getEigen(V,param ;irange = 1:3)
    ψ_0 = 1/sqrt(2) *(eig[:,1] + im * eig[:,2])

    ψ_0 = convert(Array{Complex},ψ_0)

    @show norm(ψ_0)*param.Δx

    F(t) = 0.1*sin((2)t)

    simulate(ψ_0,param,V,F)
end
function hello()
    println("Hello")
end
end
