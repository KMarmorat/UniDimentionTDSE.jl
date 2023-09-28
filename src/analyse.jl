function P_windows(ψ,H_0::SymTridiagonal,E::Real,γ::Real)
    """Compute the probability P(E,n,γ) at E"""
    H1 = H_0 + I*(-E +√(im) *γ )
    H2 = H_0 + I*(-E -√(im) *γ )

    norm(H2\H1\ψ)^2
end
