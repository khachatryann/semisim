# Theory and model equations

This document collects every equation semisim implements, its assumptions,
and where it breaks down. Units are the cm-based semiconductor convention:
cm⁻³, cm²/(V·s), F/cm, eV.

References: Sze & Ng, *Physics of Semiconductor Devices* (3rd ed.);
Pierret, *Semiconductor Device Fundamentals*; Tsividis & McAndrew,
*Operation and Modeling of the MOS Transistor* (3rd ed.); Jacoboni et al.,
*Solid-State Electron.* 20, 77 (1977); Sotoodeh et al., *J. Appl. Phys.* 87,
2890 (2000); Sah, Noyce & Shockley, *Proc. IRE* 45, 1228 (1957).

## 1. Material models (`materials.py`)

### Bandgap — Varshni equation

    Eg(T) = Eg(0) − α T² / (T + β)

| Material | Eg(0) [eV] | α [eV/K] | β [K] | Eg(300 K) [eV] |
|----------|-----------|----------|-------|-----------------|
| Si       | 1.170     | 4.73e-4  | 636   | 1.125           |
| Ge       | 0.7437    | 4.774e-4 | 235   | 0.663           |

### Effective densities of states

    Nc(T) = Nc(300) (T/300)^1.5,  Nv(T) = Nv(300) (T/300)^1.5

Si: Nc = 2.8e19, Nv = 1.04e19 cm⁻³; Ge: Nc = 1.04e19, Nv = 6.0e18 cm⁻³ (Sze).

### Intrinsic carrier concentration

    ni(T) = sqrt(Nc Nv) exp(−Eg / 2kT)

**Note:** with these Nc/Nv values, ni(Si, 300 K) ≈ 6.1e9 cm⁻³. Literature
values range 1.0–1.5e10 cm⁻³ depending on the effective-mass data used;
all derived quantities (Vbi, Is, φF) are consistent within the model but can
differ from handbook numbers by the corresponding logarithmic/quadratic
factors.

### Mobility

    μ(T) = μ(300) (T/300)^a,   a_n = −2.4, a_p = −2.2 (lattice scattering)

Si: μn = 1350, μp = 480; Ge: μn = 3900, μp = 1900; GaAs: μn = 8500,
μp = 400 cm²/(V·s) (lightly-doped bulk values, used when no doping is given).

**Doping dependence (v2, opt-in).** Passing a doping `N` to
`mobility_n/p(T, N)` applies the Caughey-Thomas model before the
temperature power law:

    μ(N, T) = [ μmin + (μmax − μmin) / (1 + (N/Nref)^α) ] · (T/300)^a

| Material | carrier | μmin | Nref [cm⁻³] | α | source |
|----------|---------|------|-------------|---|--------|
| Si | e | 92 | 1.3e17 | 0.91 | Jacoboni 1977 |
| Si | h | 48 | 2.35e17 | 0.88 | Jacoboni 1977 |
| Ge | e | 850 | 2.6e17 | 0.56 | approx. fit |
| Ge | h | 300 | 1.0e17 | 1.0 | approx. fit |
| GaAs | e | 500 | 6.0e16 | 0.394 | Sotoodeh 2000 |
| GaAs | h | 20 | 1.48e17 | 0.38 | Sotoodeh 2000 |

**Simplification:** the temperature scaling multiplies the whole curve; real
ionized-impurity scattering has a different (weaker/opposite) temperature
dependence, so heavily-doped results away from 300 K are approximate.

### Einstein relation

    D = μ kT/q

Valid for non-degenerate (Boltzmann) statistics — consistent with the
degenerate-doping warning at N > 1e19 cm⁻³.

## 2. PN junction diode (`devices/diode.py`)

Abrupt junction, full-depletion approximation, Boltzmann statistics.

### Built-in potential

    Vbi = (kT/q) ln(Na Nd / ni²)

### Depletion region (applied bias V, positive = forward)

    W(V)  = sqrt( 2 ε (Vbi − V) (Na + Nd) / (q Na Nd) )
    xp Na = xn Nd            (charge neutrality)
    Cj(V) = ε A / W(V)       (depletion capacitance)

Valid only for V < Vbi (the code raises `ValueError` at or beyond Vbi, where
the approximation collapses). High forward bias also violates the
low-level-injection assumption well before Vbi.

### Saturation current (long-base ideal diode)

    Is = q A ni² ( Dn / (Ln Na) + Dp / (Lp Nd) ),   L = sqrt(D τ)

Assumes quasi-neutral regions much longer than the diffusion lengths
(long-base), low-level injection, and no recombination in the depletion
region. With `mobility_model="constant"` (default) minority-carrier
mobilities are the lightly-doped bulk values; `"caughey-thomas"` evaluates
μn at Na (electrons diffuse through the p side) and μp at Nd.

### I-V characteristic

    I = Is ( exp(V / n VT) − 1 ),   VT = kT/q

`n` is a user-set ideality factor (default 1). With series resistance Rs the
implicit equation

    I = Is ( exp((V − I Rs) / n VT) − 1 )

has the exact closed-form solution (Banwell & Jayakumar, 2000):

    I = (n VT / Rs) · W₀[ (Is Rs / n VT) · exp((V + Is Rs) / n VT) ] − Is

where W₀ is the principal branch of the Lambert W function. semisim
evaluates W₀ from the *logarithm* of its argument (`_math.lambertw_log`),
so the expression never overflows even at V ≫ 1 V.

### Recombination-generation current (v2, opt-in `tau_rg=`)

Sah-Noyce-Shockley space-charge current with an effective lifetime τ_rg
(midgap traps, τn = τp):

    I_rg = I_r0(V) ( exp(V / 2VT) − 1 ),   I_r0 = q A ni W / (2 τ_rg)

- Reverse bias: I_rg → −q A ni W(V) / 2τ_rg, the generation current, which
  grows as sqrt(Vbi − V) and dominates Si leakage at room temperature.
- Low forward bias: the n = 2 component that bends the log I-V.
- W is evaluated at min(V, 0): bias-dependent in reverse, frozen at the
  equilibrium width in forward (the depletion formula fails near Vbi; this
  slightly overestimates forward recombination).

### Reverse breakdown (v2, opt-in `breakdown_voltage=`)

SPICE-style exponential knee added to the junction current:

    I_bv = −Ibv exp( −(V + BV) / (n_bv VT) )

so |I| = Ibv (default 1 mA) at V = −BV. `breakdown_voltage="auto"` uses the
Sze & Gibbons universal estimate for abrupt junctions:

    BV ≈ 60 (Eg/1.1)^1.5 (N_B / 1e16)^−0.75  V,   N_B = Na Nd / (Na + Nd)

(±20 % for 1e14–1e18 cm⁻³; below ~5 V real junctions break down by
tunneling, which the knee only mimics empirically).

### Solving with all terms and series resistance

When R-G or breakdown is enabled, `V = Vj + Rs I(Vj)` is solved for the
junction voltage Vj by vectorized bisection (`_math.bisect_monotonic`) —
I(Vj) is strictly increasing, so the root is bracketed by [min(V,0),
max(V,0)] and convergence to machine precision is guaranteed. The pure
Shockley case keeps the exact Lambert-W solution.

### Small-signal conductance (v2)

    g = gj / (1 + Rs gj),   gj = dI/dVj   (analytic, all terms)

`1/g` is the dynamic resistance; at zero bias g = Is/VT (ideal case).

**Not modeled:** high-level injection, band-gap narrowing, short-base
diodes, diffusion capacitance, series resistance of the neutral regions
computed from geometry (Rs is a lumped user parameter).

### Band diagram

Depletion approximation with the p-side Fermi level as energy reference:

    p neutral:  Ev = −kT ln(Nv/Na),  Ec = Ev + Eg
    φ(x) parabolic in the depletion region, total bending = Vbi − V

Under bias the electron/hole quasi-Fermi levels are drawn flat and split by
qV across the junction. This is the standard textbook illustration; it is
**not** a self-consistent drift-diffusion solution.

## 3. MOSFET (`devices/mosfet.py`)

Long-channel enhancement MOSFET (NMOS or PMOS), gradual-channel
approximation, square-law charge model.

### PMOS (v2, `channel="p"`)

All equations below are written for NMOS. A PMOS is evaluated as its mirror
image: body doping N is donors, the default gate is p⁺ poly
(Vfb = +(Eg/2 + φF)), and

    Vth = Vfb − 2φF − γ sqrt(2φF − Vsb)

The current is computed in NMOS coordinates (−Vgs, −Vds, −Vsb) and returned
with a negative sign (positive current flows *into* the drain). The default
mobility is the hole mobility. `body_doping` replaces v1's `na_substrate`
(still accepted, with a `DeprecationWarning`).

### Electrostatics

    Cox  = ε_ox / tox                       (ε_ox = 3.9 ε0)
    φF   = VT ln(Na / ni)                    (bulk Fermi potential, p-body)
    γ    = sqrt(2 q ε_s Na) / Cox            (body-effect coefficient)
    Vfb  = −(Eg/2 + φF)                      (ideal n+ poly gate, zero oxide
                                              charge; user-overridable)
    Vth  = Vfb + 2φF + γ sqrt(2φF + Vsb)

### Drain current (square law)

    k = μ Cox W/L,   Vov = Vgs − Vth

    Triode      (Vds < Vov):  Id = k (Vov − Vds/2) Vds · (1 + λ Vds)
    Saturation  (Vds ≥ Vov):  Id = (k/2) Vov² · (1 + λ Vds)

The channel-length-modulation factor is applied in **both** regions so the
I-V curve is continuous at Vds = Vov (a common SPICE Level-1 convention;
physically λ only acts in saturation).

### Subthreshold conduction

With `subthreshold=True` (default) the overdrive is replaced by an
EKV-style softplus interpolation:

    Vov_eff = 2 m VT ln(1 + exp((Vgs − Vth) / 2 m VT)),
    m = 1 + γ / (2 sqrt(2φF + Vsb))

- Strong inversion (Vgs ≫ Vth): Vov_eff → Vgs − Vth (square law recovered).
- Weak inversion (Vgs ≪ Vth): Id ∝ exp((Vgs − Vth)/m VT), i.e. the standard
  subthreshold slope SS = m VT ln 10 (≈ 76 mV/dec for the default device).

**Simplifications:** the interpolation is smooth and has the correct
asymptotes but is not a self-consistent charge-sheet solution; the
subthreshold current has no (1 − exp(−Vds/VT)) drain-saturation factor; the
strong-inversion prefactor omits the 1/m factor of full EKV. Errors near
Vgs ≈ Vth are of order 10–20%, which is inherent to any square-law-level
model.

### Velocity saturation (v2, opt-in `vsat=`)

Lateral-field mobility degradation μ_eff = μ / (1 + E/Ec) with
Ec = 2 vsat / μ gives (Sze & Ng §6.2.4):

    Triode:  Id = k (Vov − Vds/2) Vds / (1 + Vds/(Ec L)) · (1 + λ Vds)
    Vdsat  = Vov · Ec L / (Vov + Ec L)
    Saturation: the triode expression at Vds = Vdsat, times (1 + λ Vds)

Limits: Ec L → ∞ recovers the square law; Ec L → 0 gives
Id → W Cox vsat Vov (linear in overdrive), the short-channel limit.

### Small-signal parameters (v2)

`small_signal(vgs, vds, vsb)` returns gm = ∂Id/∂Vgs, gds = ∂Id/∂Vds,
gmb = ∂Id/∂Vbs and ro = 1/gds from central differences of the (smooth)
model. All are reported as positive magnitudes for both polarities.

**Not modeled:** mobility degradation with vertical field, DIBL, Vth
roll-off, gate leakage.

## 4. Validity summary

| Quantity | Trustworthy | Degrades |
|----------|-------------|----------|
| Vbi, W, Cj | Non-degenerate doping, |V| ≪ breakdown | Doping > 1e19; V → Vbi |
| Diode I-V | Moderate forward bias; low bias & reverse with `tau_rg` | High injection; BV estimate ±20 %, Zener regime empirical |
| Vth, γ, φF | Long channel, clean oxide | Real oxide charge/poly depletion (use `vfb=`) |
| MOSFET Id | Vov ≳ 5 VT or Vgs ≲ Vth − 5 VT; L ≳ 0.5 µm (with `vsat`) | Near threshold; DIBL-dominated short channels |
