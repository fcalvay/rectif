#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 23:08:44 2025

@author: florentcalvayrac
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Redressement CA • Diode / Pont + Filtre RC", layout="wide")
st.title("🔌 Redressement CA — 1 diode / Pont à 4 diodes + filtre RC")

# ======================
# Sidebar : paramètres
# ======================
with st.sidebar:
    st.header("Source & Transformateur")
    Vprim_rms = st.slider("Tension primaire (V RMS)", 10.0, 260.0, 230.0, 1.0)
    f = st.slider("Fréquence (Hz)", 10.0, 200.0, 50.0, 1.0)
    Np = st.slider("Spire primaire Np", 50, 5000, 1000, 10)
    Ns = st.slider("Spire secondaire Ns", 5, 5000, 200, 5)
    Rw_per_100 = st.slider("Résistance cuivre secondaire (mΩ / 100 spires)", 10.0, 800.0, 120.0, 5.0)

    st.header("Redressement")
    mode = st.selectbox("Mode", ["Demi-onde (1 diode)", "Pont (4 diodes)"])
    Vd = st.slider("Chute directe diode Vd (V)", 0.2, 1.2, 0.7, 0.05)
    rd = st.slider("Résistance dynamique diode rd (Ω)", 0.0, 2.0, 0.05, 0.01)

    st.header("Filtre & Charge")
    Rload = st.slider("Résistance de charge R (Ω)", 1.0, 5000.0, 200.0, 1.0)
    C_uF = st.slider("Capacité C (µF)", 0.0, 10000.0, 470.0, 10.0)
    C = C_uF * 1e-6

    st.header("Simulation")
    cycles = st.slider("Nombre de périodes simulées", 4, 30, 8, 1)
    oversamp = st.slider("Sur-échantillonnage par période", 200, 4000, 2000, 100)

# ======================
# Grandeurs dérivées
# ======================
w = 2*np.pi*f
T = 1.0 / f
t = np.linspace(0, cycles*T, cycles*oversamp, endpoint=False)
dt = t[1] - t[0]

# Transfo : rapport de spires & résistance de bobinage secondaire ~ Ns
Vs_rms = Vprim_rms * (Ns / max(Np, 1))
Vs_peak = np.sqrt(2) * Vs_rms
v_sec = Vs_peak * np.sin(w*t)  # tension secondaire vraie (avec signe)

Rw_sec = (Rw_per_100/1000.0) * (Ns/100.0)  # Ω

# Diodes en série selon mode
if mode.startswith("Pont"):
    n_diodes = 2  # deux diodes en série à chaque alternance
    rect_source = np.abs(v_sec)  # schéma équivalent côté secondaire
    # pour le calcul de puissance d'entrée, on gardera v_sec & i_sec avec signe
else:
    n_diodes = 1
    rect_source = np.maximum(v_sec, 0.0)  # demi-onde

Vd_eff = n_diodes * Vd
rd_eff = n_diodes * rd
R_series = Rw_sec + rd_eff  # résistance de conduction (source -> capa)

# ======================
# Boucle d’intégration
# Modèle charge RC en // : dv/dt = (i_diode - v/R)/C
# Conduction si Vth > vcap (avec Vth = vrect - Vd_eff)
# ======================
v_cap = np.zeros_like(t)
i_diode = np.zeros_like(t)
i_sec = np.zeros_like(t)     # courant secondaire (signe respecté pour puissance entrée)

for k in range(1, len(t)):
    v_src_rect = rect_source[k]
    Vth = v_src_rect - Vd_eff

    # Test conduction (approx diode idéale + rd, avec série R_series)
    if Vth > v_cap[k-1] and R_series > 0:
        i_cond = (Vth - v_cap[k-1]) / R_series  # A
        if i_cond < 0:
            i_cond = 0.0
    else:
        i_cond = 0.0

    # Charge // : i_C = C dv/dt = i_diode - v/R
    if C > 0:
        dv = ((i_cond - (v_cap[k-1]/Rload))) * dt / C
        v_cap[k] = max(0.0, v_cap[k-1] + dv)
    else:
        # pas de capa → simple redressement + R
        if i_cond > 0:
            # tension sortie égale à (Vsrc - Vd_eff) * R/(R+R_series)
            v_cap[k] = (v_src_rect - Vd_eff) * (Rload / (Rload + R_series))
        else:
            v_cap[k] = 0.0 if mode.startswith("Demi") else 0.0

    i_diode[k] = i_cond

    # Courant secondaire i_sec (pour puissance d'entrée)
    if mode.startswith("Pont"):
        # i_sec est du signe de la tension sinusoïdale, amplitude = i_cond
        i_sec[k] = np.sign(v_sec[k]) * i_cond
    else:
        # demi-onde : courant seulement en alternances positives
        i_sec[k] = i_cond if v_sec[k] > 0 else 0.0

# ======================
# Puissances & métriques
# ======================
# Puissance entrée (secondaire transfo)
p_in = v_sec * i_sec
P_in_avg = np.trapz(p_in, t) / t[-1] if t[-1] > 0 else 0.0

# Puissance charge
p_load = (v_cap**2) / Rload
P_load_avg = np.trapz(p_load, t) / t[-1] if t[-1] > 0 else 0.0

eta = (P_load_avg / P_in_avg) if P_in_avg > 1e-12 else np.nan
Vdc = np.trapz(v_cap, t) / t[-1] if t[-1] > 0 else 0.0
Vpp = np.max(v_cap[int(0.5*len(t)):]) - np.min(v_cap[int(0.5*len(t)):])  # ondulation fin de simu
Idiode_rms = np.sqrt(np.trapz(i_diode**2, t) / t[-1]) if t[-1] > 0 else 0.0

# Tension "après diodes" idéale (pour visualiser l'avant-filtrage)
v_after_diode = np.maximum(rect_source - Vd_eff, 0.0)

# ======================
# Affichages
# ======================
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Vs secondaire (RMS)", f"{Vs_rms:.2f} V")
c2.metric("Rw secondaire", f"{Rw_sec:.3f} Ω")
c3.metric("Vdc sortie", f"{Vdc:.2f} V")
c4.metric("Ondulation ΔVpp", f"{Vpp:.2f} V")
c5.metric("η rendement", "—" if not np.isfinite(eta) else f"{100*eta:.1f} %")

c6,c7,c8 = st.columns(3)
c6.metric("I_diode,rms", f"{Idiode_rms:.3f} A")
c7.metric("P_in moyenne", f"{P_in_avg:.2f} W")
c8.metric("P_load moyenne", f"{P_load_avg:.2f} W")

st.markdown("---")

tab1, tab2 = st.tabs(["Formes d'onde", "Spectre & rappels"])

with tab1:
    fig, axs = plt.subplots(3, 1, figsize=(8,7), sharex=True)
    axs[0].plot(t, v_sec, label="v_sec (AC)")
    axs[0].plot(t, v_after_diode, label="v après diode(s)")
    axs[0].plot(t, v_cap, label="v_out filtrée")
    axs[0].set_ylabel("Tension (V)")
    axs[0].grid(True); axs[0].legend()

    axs[1].plot(t, i_diode, label="i_diode")
    axs[1].plot(t, i_sec, label="i_secondaire", alpha=0.7)
    axs[1].set_ylabel("Courant (A)")
    axs[1].grid(True); axs[1].legend()

    axs[2].plot(t, p_in, label="p_in = v_sec·i_sec")
    axs[2].plot(t, p_load, label="p_load = v_out²/R")
    axs[2].set_ylabel("Puissance (W)")
    axs[2].set_xlabel("Temps (s)")
    axs[2].grid(True); axs[2].legend()

    st.pyplot(fig, clear_figure=True)

with tab2:
    st.markdown("""
**Idées clés :**
- **Demi-onde** : une seule diode → conduction une alternance sur deux → plus d’ondulation.
- **Pont** : deux diodes en série à chaque alternance (deux chutes \(V_d\)) → conduction sur **toutes** les alternances → ondulation réduite.
- **Filtre RC** : le condensateur se **charge** quand \(v_{\text{source}} - n\_d V_d > v_{\text{out}}\) et se **décharge** via \(R\) entre les crêtes.  
  L’ondulation diminue si \(C↑\) et/ou \(R↑\), mais l’ondulation augmente si \(f↓\).
- **Rendement** : \(\eta = \overline{P_{\text{load}}}/\overline{P_{\text{in}}}\). Les pertes viennent de \(R_w\) (cuivre), des diodes (\(V_d, r_d\)), et de l’ondulation (courants pulsés).
- **Lien spires ↔ résistance** : on modélise \(R_w \propto N_s\). D’où l’intérêt de fil plus gros pour les forts courants.
""")

    st.markdown("**Astuce de dimensionnement rapide (pont + filtre C||R) :**")
    st.latex(r"""
    V_{\text{out,DC}} \approx V_{\text{p}} - 2V_d - \Delta V/2,\quad
    \Delta V \approx \frac{I_{\text{load}}}{f_{\text{ripple}}\,C},
    \quad f_{\text{ripple}}=\begin{cases}
      f & \text{(demi-onde)}\\
      2f & \text{(pont)}
    \end{cases}
    """)
    st.caption("Le simulateur calcule tout ça pas à pas, avec la résistance série et la dynamique de conduction.")

# ======================
# Fin
# ======================