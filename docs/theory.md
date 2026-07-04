# Theory and model equations

This document collects every equation semisim implements, its assumptions,
and where it breaks down. Units are the cm-based semiconductor convention:
cm⁻³, cm²/(V·s), F/cm, eV.

References: Sze & Ng, *Physics of Semiconductor Devices* (3rd ed.);
Pierret, *Semiconductor Device Fundamentals*; Tsividis & McAndrew,
*Operation and Modeling of the MOS Transistor* (3rd ed.).

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

Si: μn = 1350, μp = 480; Ge: μn = 3900, μp = 1900 cm²/(V·s).
**Simplification:** no doping dependence (Caughey-Thomas not implemented);
values are lightly-doped bulk values.

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
region. Minority-carrier mobilities are approximated by majority values.

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

**Not modeled:** avalanche/Zener breakdown, space-charge recombination
current (the n≈2 low-bias component), high-level injection, series
resistance of the neutral regions computed from geometry (Rs is a lumped
user parameter).

### Band diagram

Depletion approximation with the p-side Fermi level as energy reference:

    p neutral:  Ev = −kT ln(Nv/Na),  Ec = Ev + Eg
    φ(x) parabolic in the depletion region, total bending = Vbi − V

Under bias the electron/hole quasi-Fermi levels are drawn flat and split by
qV across the junction. This is the standard textbook illustration; it is
**not** a self-consistent drift-diffusion solution.

## 3. MOSFET (`devices/mosfet.py`)

Long-channel enhancement NMOS, gradual-channel approximation, square-law
charge model.

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

**Not modeled:** velocity saturation, mobility degradation with vertical
field, DIBL, Vth roll-off, gate leakage, PMOS (v1 is NMOS-only).

## 4. Validity summary

| Quantity | Trustworthy | Degrades |
|----------|-------------|----------|
| Vbi, W, Cj | Non-degenerate doping, |V| ≪ breakdown | Doping > 1e19; V → Vbi |
| Diode I-V | Moderate forward bias, ideal reverse | Low bias (n≈2 region), high injection, breakdown |
| Vth, γ, φF | Long channel, clean oxide | Real oxide charge/poly depletion (use `vfb=`) |
| MOSFET Id | Vov ≳ 5 VT or Vgs ≲ Vth − 5 VT | Near threshold; short channels |
