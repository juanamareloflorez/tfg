# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:51:00 2026

Monte Carlo 3D de atrapamiento resonante, Ar 106.7 nm (J=1), redistribución
completa de frecuencia (CFR), losa infinita en x,y de espesor H en z
(paredes absorbentes en z=0 y z=H).

Finales posibles de cada fotón:  escape por z=0  |  escape por z=H
(con p_exc = 0 no hay conversión a excímero)

Modo DOPPLER_ONLY: perfil Doppler puro (gamma_L = 0) y comparación de la tasa de
escape por generación g = beta/gamma con Holstein, Phys. Rev. 83, 1159 (1951):
    (6.2)  g = 1.875 / [k0 L (pi log(k0 L))^(1/2)]

con k0 = coeficiente de absorción en el centro de la línea Doppler y L = H.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.special import voigt_profile

# ------------------------------------------------------------------ parametros
P_BAR   = 1       # presion de Ar (bar)  (para la imagen de la trayectoria del foton individual)
T       = 300.0      # K
H       = 50e-6      # espesor de la losa (m); infinita en x,y
K_EXC   = 0       # coef. tres cuerpos -> Ar2*  (0 => p_exc = 0)
SEED    = None

# --- comparacion con Holstein (1951) --------------------------------------------
DOPPLER_ONLY = True    # True: perfil Doppler puro (gamma_L = 0); False: Voigt del Ar
P_SCAN       = (0.01, 0.1, 0.5, 1.0)   # barrido en presion (bar) => barrido en k0*L
N_MAX        = 40000   # fotones maximos por punto
PRESUPUESTO  = 3e8     # pasos fotonicos aprox. por punto (limita N cuando g es muy pequeño)
N0_FRAC      = 2.0     # se ajusta la cola n >= N0_FRAC/g (descarta el transitorio de modos altos)

c, kB, amu = 2.998e8, 1.381e-23, 1.6605e-27
M       = 39.948 * amu
lam     = 106.7e-9
nu0     = c / lam
A       = 1.3e8            # s^-1
gu_gl   = 3.0
re      = 2.818e-15
# fuerza de oscilador derivada de A (relacion de Einstein)
f       = A * gu_gl * c / (8 * np.pi**2 * re * nu0**2)

sigma_D = nu0 / c * np.sqrt(kB * T / M)     # Hz, desviacion estandar de la gaussiana doppler

def dens(p_bar):                            # m^-3
    return p_bar * 1e5 / (kB * T)

def params(p_bar):
    N = dens(p_bar)
    gam = 0.5 * (A / (2*np.pi) + 3*re*c*lam*f*N/(8*np.pi))   # HWHM lorentz (Hz)
    if DOPPLER_ONLY:
        gam = 0.0                                              # sin ensanchamiento lorentziano
    k0 = N * lam**2 / (8*np.pi) * gu_gl * A                    # m^-1 * Hz
    p_exc = K_EXC * (N*1e-6)**2 / (A + K_EXC * (N*1e-6)**2)    # prob. por absorcion
    return N, gam, k0, p_exc


def _voigt_draw(rng, n, gam):
    return sigma_D*rng.standard_normal(n) + gam*np.tan(np.pi*(rng.random(n)-0.5))

def sample_freq(rng, n, gam):
    """Voigt = gaussiana (sigma_D) + lorentziana (gam)""" "aqui solo se usa la lorentziana"
    x = _voigt_draw(rng, n, gam)
    return x

def kappa(x, gam, k0):
    return k0 * voigt_profile(x, sigma_D, gam)

def _isotropic(rng, n):
    """Direcciones isotropas en 3D: (n,3) vectores unitarios."""
    cos_t = 2*rng.random(n) - 1
    sin_t = np.sqrt(1 - cos_t**2)
    phi = 2*np.pi*rng.random(n)
    return np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t], axis=1)

# ------------------------------------------------------------ un solo foton
def track_photon(rng, p_bar, max_steps=200000):
    N, gam, k0, p_exc = params(p_bar)
    pos = np.array([0.0, 0.0, rng.random()*H])      # x,y arbitrarios (losa infinita)
    pts = [pos.copy()]
    x =sample_freq(rng, 1, gam)[0]
    fate = "máx. pasos"
    for _ in range(max_steps):
        k = kappa(x, gam, k0)
        s = -np.log(1 - rng.random()) / k
        d = _isotropic(rng, 1)[0]
        new = pos + s*d
        if new[2] < 0 or new[2] > H:
            zw = 0.0 if new[2] < 0 else H
            fate = "escape z=0" if zw == 0.0 else "escape z=H"
            t = (zw - pos[2]) / d[2]                # recorta el segmento en la pared
            pts.append(pos + t*d)
            break
        pos = new
        pts.append(pos.copy())
        if rng.random() < p_exc:
            fate = "excímero"
            break
        x = sample_freq(rng, 1, gam)[0]             # CFR
    return np.array(pts), fate

# ------------------------------------------------------------ ensemble
def ensemble(rng, p_bar, n=1000, max_steps=100000):
    """Solo hace falta z (la losa es infinita en x,y)."""
    N, gam, k0, p_exc = params(p_bar)
    z = rng.random(n) * H
    x = np.zeros(n) if START_AT_NU0 else sample_freq(rng, n, gam)
    alive = np.ones(n, bool)
    fate = np.zeros(n, int)            # 0 vivo, 1 excimero, 2 escape z=0, 3 escape z=H
    nabs = np.zeros(n, int)
    for _ in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        k = kappa(x[idx], gam, k0)
        s = -np.log(1 - rng.random(idx.size)) / k
        z[idx] += s*(2*rng.random(idx.size) - 1)    # cos(theta) uniforme en [-1,1]
        lo = z[idx] < 0
        hi = z[idx] > H
        fate[idx[lo]] = 2
        fate[idx[hi]] = 3
        alive[idx[lo | hi]] = False
        idx = idx[~(lo | hi)]
        nabs[idx] += 1
        conv = rng.random(idx.size) < p_exc
        fate[idx[conv]] = 1
        alive[idx[conv]] = False
        idx = idx[~conv]
        x[idx] = sample_freq(rng, idx.size, gam)
    return fate, nabs

# ------------------------------------------------------------ comparacion con Holstein (1951)
def k0L(p_bar):
    """k0*L de Holstein: coeficiente de absorcion en el centro de la linea x espesor de la losa."""
    N, gam, k0, p_exc = params(p_bar)
    return H * kappa(0.0, gam, k0)

def g_holstein_62(K):
    """Holstein 1951, Ec. (6.2), losa infinita, Doppler:
    g = 1.875 / [k0 L (pi log(k0 L))^(1/2)]."""
    return 1.875 / (K * np.sqrt(np.pi * np.log(K)))

def estimar_g(nabs, n0):
    """Con p_exc = 0 el nº de absorciones n es asintoticamente geometrico, P(n>=m) ~ (1-g)^m;
    estimador de maxima verosimilitud sobre la cola n >= n0:  g = 1/(1 + <n - n0>)."""
    t = nabs[nabs >= n0] - n0
    g = 1.0/(1.0 + t.mean())
    return g, g/np.sqrt(t.size)

def comparar_holstein(rng):
    print("Doppler puro, p_exc = 0, sin truncar.  g = beta/gamma = escape por generación")
    print(f"{'P (bar)':>8} {'k0 L':>9} {'N fot':>7} {'g MC':>11} {'± err':>9} "
          f"{'g (6.2)':>11} {'MC/6.2':>8} {'<n>':>9} {'vivos':>6}")
    for p in P_SCAN:
        K = k0L(p)
        g62 = g_holstein_62(K)
        n = int(max(2000, min(N_MAX, PRESUPUESTO*g62)))
        fate, nabs = ensemble(rng, p, n=n, max_steps=int(60/g62))
        g, e = estimar_g(nabs, int(N0_FRAC/g62))
        print(f"{p:8.0e} {K:9.0f} {n:7d} {g:11.4e} {e:9.1e} {g62:11.4e} {g/g62:8.3f} "
              f" {nabs.mean():9.1f} {int((fate==0).sum()):6d}", flush=True)

# ------------------------------------------------------------ figura
def plot_photon(pts, fate, p_bar):
    """Vista lateral (x-z) con la losa sombreada y vista superior (x-y)."""
    P = pts * 1e6                      # micras
    n = len(P) - 1
    Hum = H * 1e6
    xc, yc = P[:, 0] - P[0, 0], P[:, 1] - P[0, 1]
    z = P[:, 2]
    col = np.arange(n)

    fig, (axs, axt) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # --- vista lateral x-z
    segs = np.stack([np.column_stack([xc[:-1], z[:-1]]),
                     np.column_stack([xc[1:],  z[1:]])], 1)
    lc = LineCollection(segs, cmap="viridis", linewidths=0.9)
    lc.set_array(col)
    axs.add_collection(lc)
    axs.axhspan(0, Hum, color="0.93", zorder=0)
    axs.axhline(0, color="k", lw=1.5); axs.axhline(Hum, color="k", lw=1.5)
    xr = max(np.ptp(xc) / 2 * 1.15, Hum)
    xm = (xc.min() + xc.max()) / 2
    axs.set_xlim(xm - xr, xm + xr); axs.set_ylim(-0.12*Hum, 1.12*Hum)
    axs.plot(xc[0], z[0], "o", color="#2ca02c", ms=9, zorder=5, label="origen")
    axs.plot(xc[-1], z[-1], "X", color="#d62728", ms=11, zorder=5, label=f"final: {fate}")
    axs.set_xlabel("Δx (µm)"); axs.set_ylabel("z (µm)")
    axs.set_title(f"Vista lateral (x–z), losa de {Hum:g} µm — escala vertical ampliada")
    axs.legend(loc="upper right", fontsize=8)

    # --- vista superior x-y
    segs = np.stack([np.column_stack([xc[:-1], yc[:-1]]),
                     np.column_stack([xc[1:],  yc[1:]])], 1)
    lc2 = LineCollection(segs, cmap="viridis", linewidths=0.9)
    lc2.set_array(col)
    axt.add_collection(lc2)
    r = max(np.abs(np.concatenate([xc, yc])).max() * 1.15, 1e-3)
    axt.set_xlim(-r, r); axt.set_ylim(-r, r); axt.set_aspect("equal")
    axt.plot(0, 0, "o", color="#2ca02c", ms=9, zorder=5)
    axt.plot(xc[-1], yc[-1], "X", color="#d62728", ms=11, zorder=5)
    axt.set_xlabel("Δx (µm)"); axt.set_ylabel("Δy (µm)")
    axt.set_title("Vista superior (x–y)")
    cb = fig.colorbar(lc2, ax=axt, shrink=0.8, pad=0.02)
    cb.set_label("número de absorciones")

    fig.suptitle(f"Fotón individual, Ar {p_bar:g} bar, {n} pasos")
    plt.show()

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    N, gam, k0, p_exc = params(P_BAR)
    perfil = "Doppler puro" if DOPPLER_ONLY else "Voigt"
    print(f"Perfil: {perfil} | N = {N:.3e} m^-3 | gamma_L(HWHM) = {gam:.3e} Hz | "
          f"p_exc por absorción = {p_exc:.3e}")
    print(f"k(centro) = {kappa(0.0, gam, k0):.3e} m^-1  "
          f"(recorrido {1e9/kappa(0.0, gam, k0):.2f} nm)  |  H = {H*1e6:g} µm  |  "
          f"k0 L = {k0L(P_BAR):.4g}")

    # primer foton con al menos MIN_STEPS absorciones (trayectoria ilustrativa)
    MIN_STEPS = 100
    ntry = 0
    while ntry < 100000:
        ntry += 1
        pts, fate = track_photon(rng, P_BAR)
        if len(pts) - 1 >= MIN_STEPS:
            break
    print(f"Fotón individual (el nº {ntry} sorteado, primero con >= {MIN_STEPS} "
          f"pasos): {len(pts)-1} pasos, final = {fate}")
    plot_photon(pts, fate, P_BAR)

    if DOPPLER_ONLY:
        print(f"\nf = {f:.3f} (derivada de A)")
        comparar_holstein(rng)
