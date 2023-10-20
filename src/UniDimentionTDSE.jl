module UniDimentionTDSE
using LinearAlgebra,DelimitedFiles,BandedMatrices
export Hamiltonian,simulate,test,getEigen,test_wigner,simulate_coupled,simulate_coupled_new
include("absorbtion.jl")
include("analyse.jl")

struct SimulationParameter
    Δx::Float64
    Lmin::Float64
    Lmax::Float64
    Δt::Float64
    time::Float64
    Nt::Int64
    Neig::Int64
    Filename::String
end


function writeToFile(ψ,param::SimulationParameter,eigVecs,F,t,io::IOStream,extrafunctions...;
    lineNorm::Integer=1,numberLine::Integer=1)

    ξ = ψ[(lineNorm-1)*(end÷numberLine)+1:(lineNorm)*(end÷numberLine)]


    pop = reshape([dot(ξ,ϕ)*param.Δx for ϕ in eachcol(eigVecs)],1,param.Neig)
    ψ_norm = norm(ξ)*sqrt(param.Δx)
    if extrafunctions == ()
        writedlm(io,[t  pop F(t) ψ_norm],';')
        return
    end

    extra = transpose(collect(Iterators.flatten([f(ξ,t) for f in extrafunctions])))
    writedlm(io,[t  pop F(t) ψ_norm extra],';')

end

function writeToFile(ψ,param::SimulationParameter,F,t,io::IOStream,extrafunctions...)

    extra = transpose(collect(Iterators.flatten([f(ψ[1:end÷2],t) for f in extrafunctions])))
        ψ_norm = norm(ψ[1:end÷2])*sqrt(param.Δx)

    writedlm(io,[t  F(t) ψ_norm extra],';')

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
    @. Htop.dv = 1 - im*H.dv * Δt/2
    @. Hbottom.dv = 1 + im*H.dv * Δt/2
end

function buildCrankNicolson_coupled!(H,Htop,Hbottom,Δt)
    "Update the upper and lower part of Crank Nicolson"
    n,_ = size(H)
    n ÷= 2

    @. Htop[band(n)] =  - im * H[band(n)] * Δt/2
    @. Hbottom[band(n)] = + im * H[band(n)] * Δt/2

    n = -n
    @. Htop[band(n)] =  - im * H[band(n)] * Δt/2
    @. Hbottom[band(n)] = + im * H[band(n)] * Δt/2
end

function buildCrankNicolson(H,Δt)
    "Return the upper and lower part of Crank Nicolson"
    Htop = I - im * H * Δt/2
    Hbottom = I + im * H * Δt/2
    (Htop,Hbottom)
end

function Hamiltonian!(H::SymTridiagonal,Hdiag_0,x::AbstractRange,F,t;μ::Real=1)
    "Update the Hamiltonian"
    @. H.dv = Hdiag_0.dv + F(t)*x
end

function Hamiltonian(param::SimulationParameter,V;μ::Real=1,Type=Float64)
    x = buildx(param)
    Hamiltonian(V,x;μ,Type)
end

function Hamiltonian(x::AbstractRange,V;μ::Real=1,Type=Float64)
    "Create the time independant Hamiltonian"
    Δx = step(x)
    midline::Vector{Type} = V.(x)   .+ 1/(μ*(Δx)^2)
    topline::Vector{Type} = zero(x) .- 1/(μ*2*(Δx)^2)
    SymTridiagonal((midline),(topline))
end

function simulate(ψ,param::SimulationParameter,V
    ;μ::Real = 1)
    simulate(ψ,param,V,(t)->0;μ)
end

function simulate(ψ,param::SimulationParameter,V,F::Function,extrafunctions...
    ;μ::Real=1
    ,endTime::Real=0
    ,startTime::Real=1
    ,read_access="w"
    ,output="wavefunctions"
    ,Veigen=nothing)
    @assert (iszero(imag(param.Δt)) || iszero(real(param.Δt)==0))

    x = buildx(param)

    H = (Hamiltonian(x,V;μ,Type=ComplexF64))

    _,eigVecs = getEigen(isnothing(Veigen) ? V : Veigen,param;irange= 1:param.Neig,μ)
    simulation_loop(ψ,param,H,F,buildCrankNicolson,buildCrankNicolson!,Hamiltonian!,eigVecs,extrafunctions...
    ;μ,output,endTime,startTime,read_access)
end


function simulation_loop(ψ,param::SimulationParameter,H,F,bCNicolson,bCNicolson!,bHamilton!,eigVecs,extrafunctions...
    ;μ::Real=1
    ,lineNorm::Integer=1
    ,numberLine::Integer=1
    ,endTime::Real=0
    ,startTime::Real=1
    ,read_access="w"
    ,output="wavefunctions")

    endTime = iszero(endTime) ? param.Nt : endTime/param.Δt
    startTime = isone(startTime) ? 1 : startTime/param.Δt

    x = buildx(param)
    H_0 = copy(H)
    (Htop,Hbottom) = bCNicolson(H,param.Δt)

    @show startTime
    @show endTime
    open(param.Filename,read_access) do io
        for i = startTime:endTime
            t = i*param.Δt

            bHamilton!(H,H_0,x,F,t+param.Δt/2;μ)
            bCNicolson!(H,Htop,Hbottom,param.Δt)
            propagate!(ψ,Htop,Hbottom)

            iszero(real(param.Δt)) && begin  normalize!(ψ); ψ/=param.Δx end

            writeToFile(ψ,param,eigVecs,F,t,io,extrafunctions...;lineNorm,numberLine)
        end
    end

    open(param.Filename * "." * output,"w") do io
        writedlm(io,ψ,';')
    end
end

function rotate!(ψ1,ψ2,F,t,Δt)
    time = t + Δt/2 
    step = Δt/2
    @. ψ1 = cos(F(time)*step)*ψ1 + im*sin(F(time)*step)*ψ2
    @. ψ2 = cos(F(time)*step)*ψ2 + im*sin(F(time)*step)*ψ1
end

function Hamiltonian_coupled(x::AbstractRange,V1,V2;μ::Real=1)
    "Create the time independant Hamiltonian"
    n = length(x)
    Δx = step(x)
    midline = vcat(V1.(x),V2.(x))   .+ 1/(μ*(Δx)^2)
    topline = zeros(2n-1) .- 1/(μ*2*(Δx)^2)
    topline[n] = 0
    BandedMatrix(0=>midline,1=>topline,-1=>topline,(n)=>ones(n),(-n)=>ones(n))
end

function Hamiltonian_coupled!(H,H_0,x::AbstractRange,F,t;μ::Real=1)
    "Update the Hamiltonian"
    n = length(x)
    @. H[band(n)] = F(t)
    @. H[band(-n)] = F(t)
end

function simulate_coupled(ψ1,ψ2,param::SimulationParameter,V1,V2,F::Function,extrafunctions...
    ;μ::Real=1,lineNorm::Integer=1,numberLine::Integer=1,
    double_simulation::Bool=false,endTime::Real=0)
    @assert (iszero(imag(param.Δt)) || iszero(real(param.Δt)==0))
    x = buildx(param)

    Type = ComplexF64
    H1 = Hamiltonian(x,V1;μ,Type)
    H2 = Hamiltonian(x,V2;μ,Type)
    _,eigVecs = getEigen(V2,param;irange= 1:param.Neig,μ)


    open(param.Filename,"w") do io
        (Htop1,Hbottom1) = buildCrankNicolson(H1,param.Δt)
        (Htop2,Hbottom2) = buildCrankNicolson(H2,param.Δt)
        for i = 1:endTime/param.Δt
            t = i*param.Δt
            rotate!(ψ1,ψ2,F,t,param.Δt)

            propagate!(ψ1,Htop1,Hbottom1)
            propagate!(ψ2,Htop2,Hbottom2)

            rotate!(ψ1,ψ2,F,t,param.Δt)
            writeToFile(ψ1,param,eigVecs,F,t,io,extrafunctions...)
        end
    end

    open(param.Filename * "." * "wavefunctions","w") do io
        writedlm(io,vcat(ψ1,ψ2),';')
    end

    if double_simulation
        @show "second Simulation"
        simulate(ψ1,param,V1,t->0,extrafunctions...;μ,startTime=endTime,read_access="a",output="wavefunctions_second",Veigen=V2)
    end
end
function getEigen(V,param::SimulationParameter;irange::UnitRange=1:1,μ::Real=1)
    x = buildx(param)
    getEigen(V,x;irange,μ)
end

function getEigen(V,x::StepRangeLen;irange=1:1,μ::Real=1)

    H = Hamiltonian(x,V;μ)
    E, ψs =  eigen(H,irange)

    for ψ in eachcol(ψs)
        normalize!(ψ)
        ψ ./= sqrt(step(x))
    end

    (E,ψs)
end
function buildx(param::SimulationParameter)
    range(param.Lmin,param.Lmax;step=param.Δx)
end


function test()
    param = SimulationParameter(
    0.01,
    -10,
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
    x = buildx(param)

    
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
    -5,
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
    -5,
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
    H = Hamiltonian(x,V)
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
    -50,
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
