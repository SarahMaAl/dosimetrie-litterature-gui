#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier : dosimetrie_litterature_gui.py
Auteur  : Sarah MARTIN-ALONSO
Création: 2025-08-21
Résumé  :
    Application GUI (Tkinter) d’aide à la dosimétrie basée sur des méthodes
    de la littérature. Elle fournit :
      - Gestion des prescriptions pour PTV/OAR/PRV.
      - Conversion automatique %Rx → Gy (niveaux usuels sans DVH).
      - Estimation de marges via gradient (%/mm) : mode manuel ou basé
        sur structures (Rx×% ou absolue).
      - Génération d’anneaux (rings) avec contraintes Dmax/Dmean (TG-263 friendly).
      - Règle gradient : conversions % ↔ mm ↔ Gy.
      - Conseiller heuristique de pondération d’objectifs.
      - Checklist d’optimisation avec rappel de g ACTUEL.
      - Outils DVH facultatifs (indices ICRU 83, gEUD, Paddick) + tracé.
      - Calculette simple Dose ↔ %Rx.
      - Lint de nomenclature (rappels TG-263).

Licence : MIT
"""

from __future__ import annotations

# ----- Imports standard -----
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# ----- Imports tierce partie -----
import numpy as np
import pandas as pd

import matplotlib

# Important : choisir le backend avant pyplot
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt  # noqa: E402

import tkinter as tk  # noqa: E402
from tkinter import ttk, filedialog, messagebox  # noqa: E402

# ============================================================================
# Constantes module
# ============================================================================

APP_TITLE: str = "Outil de dosimétrie – méthodes de la littérature"
APP_GEOMETRY: str = "1420x940"

DEFAULT_GRADIENT: float = 6.0            # g [%/mm]
DEFAULT_BOUNDARY_DELTA: float = 1.0       # δ [mm] pour Dmax(r_in+δ)
DEFAULT_PREFIX_CHOICES: Tuple[str, str] = ("z", "_")
DEFAULT_PTV_STYLE: str = "cGy"            # "Relatif" ou "cGy"
RING_DEFAULTS_MM: Tuple[str, str, str] = ("0-5", "5-10", "10-20")

DVH_COLS_STANDARD: Tuple[str, str, str] = ("Structure", "DoseGy", "Vrel")

# Niveaux %Rx usuels pour l’onglet "%Rx → Gy (auto)"
DEFAULT_PCT_LEVELS: Tuple[float, ...] = (0.80, 0.90, 0.95, 0.98, 1.00, 1.05, 1.07, 1.10)

# ============================================================================
# Modèle & utilitaires
# ============================================================================


@dataclass
class Structure:
    """
    Représentation d’une structure (ROI) de planification.

    :param name: Nom de la structure (conforme à la nomenclature locale).
    :param structure_type: Type logique ('PTV', 'OAR', 'PRV').
    :param rx: Prescription en Gy (si applicable, p.ex. PTV).
    :param a_param: Paramètre « a » pour gEUD (négatif pour cibles, positif OAR).
    """
    name: str
    structure_type: str  # 'PTV' | 'OAR' | 'PRV'
    rx: Optional[float] = None  # Gy
    a_param: Optional[float] = None  # gEUD paramètre 'a'


def format_value(x: object, digits: int = 3) -> str:
    """
    Formate une valeur numérique avec sécurité (NaN/Inf → tiret cadratin).

    :param x: Valeur à formater.
    :param digits: Nombre de décimales si numérique.
    :return: Chaîne formatée.
    """
    if x is None:
        return "—"
    try:
        if isinstance(x, (float, int, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return "—"
        return f"{x:.{digits}f}" if isinstance(x, (float, int, np.floating)) else str(x)
    except Exception:
        return str(x)


def pad_mm(x: float) -> str:
    """
    Formate une distance en mm dans un style compact et parseable.

    Si entier → 2 chiffres ("03", "07"). Si décimal → « 2p5 » pour 2.5 mm.

    :param x: Distance en mm.
    :return: Représentation compacte.
    """
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x)):02d}"
    s = f"{x:.1f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def strip_accents_keep_ascii(text: str) -> str:
    """
    Normalise une chaîne en ASCII sans accents ; remplace espaces/tirets par '_'.

    :param text: Texte source.
    :return: Texte ASCII sans accents et avec underscores.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(" ", "_").replace("-", "_")
    return text


# ============================================================================
# DVH : primitives optionnelles
# ============================================================================


def _prep_dvh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un DataFrame DVH (cumulatif) avec colonnes standardisées.

    Colonnes reconnues (case-insensitive) :
      - Structure | ROI | Name → "Structure"
      - DoseGy | Dose → "DoseGy"
      - Vrel | Volume | V → "Vrel"

    :param df: Données DVH en entrée.
    :return: DataFrame avec colonnes (Structure, DoseGy, Vrel), types numériques,
             trié, et Vrel borné à [0, 1].
    """
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}
    struct_col = cols.get("structure") or cols.get("roi") or cols.get("name") or "Structure"
    dose_col = cols.get("dosegy") or cols.get("dose") or "DoseGy"
    v_col = cols.get("vrel") or cols.get("volume") or cols.get("v") or "Vrel"
    d = d.rename(columns={struct_col: "Structure", dose_col: "DoseGy", v_col: "Vrel"})
    d = d[list(DVH_COLS_STANDARD)].dropna()
    d["DoseGy"] = pd.to_numeric(d["DoseGy"], errors="coerce")
    d["Vrel"] = pd.to_numeric(d["Vrel"], errors="coerce")
    d = d.dropna().sort_values(["Structure", "DoseGy"], ascending=[True, True])
    d = d[(d["Vrel"] >= 0) & (d["Vrel"] <= 1)]
    return d


def d_at_v(dvh: pd.DataFrame, v_target: float) -> Optional[float]:
    """
    Interpole la dose D à volume relatif v_target sur DVH cumulatif.

    :param dvh: DataFrame avec colonnes "DoseGy", "Vrel" (cumulatif).
    :param v_target: Volume relatif (0–1).
    :return: Dose [Gy] estimée ou None si indisponible.
    """
    if dvh.empty:
        return None
    v = dvh["Vrel"].values
    d = dvh["DoseGy"].values
    # Garantir monotonie décroissante de Vrel avec D croissante
    v_mono = np.maximum.accumulate(v[::-1])[::-1]
    idx = np.searchsorted(-v_mono, -v_target) - 1
    idx = np.clip(idx, 0, len(d) - 2)
    v1, v2 = v_mono[idx], v_mono[idx + 1]
    d1, d2 = d[idx], d[idx + 1]
    if v2 == v1:
        return float(d1)
    return float(d1 + (v_target - v1) * (d2 - d1) / (v2 - v1))


def v_at_d(dvh: pd.DataFrame, d_target: float) -> Optional[float]:
    """
    Interpole le volume relatif V à dose d_target sur DVH cumulatif.

    :param dvh: DataFrame avec colonnes "DoseGy", "Vrel" (cumulatif).
    :param d_target: Dose [Gy] cible.
    :return: Volume relatif (0–1) estimé ou None.
    """
    if dvh.empty:
        return None
    v = dvh["Vrel"].values
    d = dvh["DoseGy"].values
    idx = np.searchsorted(d, d_target) - 1
    idx = np.clip(idx, 0, len(d) - 2)
    d1, d2 = d[idx], d[idx + 1]
    v1, v2 = v[idx], v[idx + 1]
    if d2 == d1:
        return float(v1)
    return float(v1 + (d_target - d1) * (v2 - v1) / (d2 - d1))


def geud_from_cum_dvh(dvh: pd.DataFrame, a: float) -> Optional[float]:
    """
    Calcule la gEUD à partir d’un DVH cumulatif.

    :param dvh: DataFrame (DoseGy, Vrel cumulatif décroissant).
    :param a: Paramètre 'a' de Niemierko (a<0 cible, a>0 OAR). Si a≈0,
              utilise le cas limite (moyenne géométrique).
    :return: gEUD [Gy] ou None.
    """
    if dvh.empty:
        return None
    v = np.clip(dvh["Vrel"].values, 0.0, 1.0)
    d = dvh["DoseGy"].values
    v_next = np.append(v[1:], 0.0)
    w = v - v_next
    w = np.where(w < 0, 0, w)
    s = w.sum()
    if s <= 0:
        return None
    w = w / s
    if abs(a) < 1e-9:
        eps = 1e-6
        return float(np.exp(np.sum(w * np.log(d + eps))))
    return float(np.power(np.sum(w * np.power(d, a)), 1.0 / a))


def paddick_ci(tv_cc: float, piv_cc: float, intersect_cc: float) -> Optional[float]:
    """
    Calcule l’indice de Paddick.

    :param tv_cc: Volume cible (cm³).
    :param piv_cc: Volume isodose prescrit (cm³).
    :param intersect_cc: Volume (TV ∩ PIV) (cm³).
    :return: Conformity Index (0–1) ou None si entrées invalides.
    """
    if tv_cc <= 0 or piv_cc <= 0:
        return None
    return float((intersect_cc ** 2) / (tv_cc * piv_cc))


def hi_icru(d2: Optional[float], d98: Optional[float], rx: Optional[float]) -> Optional[float]:
    """
    Calcule l’indice d’homogénéité ICRU 83 : HI = (D2% - D98%) / Rx.

    :param d2: D2% [Gy].
    :param d98: D98% [Gy].
    :param rx: Prescription Rx [Gy].
    :return: HI [—] ou None si non calculable.
    """
    if d2 is None or d98 is None or not rx or rx == 0:
        return None
    return float((d2 - d98) / rx)


# ============================================================================
# Application GUI
# ============================================================================


class DosimApp(tk.Tk):
    """
    Application Tkinter regroupant les fonctionnalités d’aide à la dosimétrie.

    Onglets principaux :
      - Prescriptions, %Rx → Gy, Marges, Anneaux, Ruler Gradient,
        Pondération, Checklist, Calculette, Nomenclature Lint, DVH.

    Notes :
      - Les valeurs de gradient g [%/mm] sont centralisées (variable Tk).
      - Les noms de rings sont compatibles TG-263 via un préfixe configurable.
      - Les fonctions DVH sont optionnelles (CSV cumulatif).
    """

    def __init__(self) -> None:
        """Initialise la fenêtre principale, l’état et construit l’UI."""
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_GEOMETRY)

        # État applicatif
        self.structures: List[Structure] = []
        self.dvh_df: Optional[pd.DataFrame] = None

        self.gradient_default = tk.DoubleVar(value=DEFAULT_GRADIENT)   # g [%/mm]
        self.boundary_delta = tk.DoubleVar(value=DEFAULT_BOUNDARY_DELTA)  # δ [mm]
        self.prefix_choice = tk.StringVar(value=DEFAULT_PREFIX_CHOICES[0])
        self.ptv_style = tk.StringVar(value=DEFAULT_PTV_STYLE)  # 'Relatif' ou 'cGy'

        # Construction UI
        self._build_ui()

        # Rafraîchir la checklist au changement de g
        self.gradient_default.trace_add("write", lambda *a: self._render_steps(self.frm_steps))

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        """Crée le notebook et instancie les onglets."""
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.frm_presc = ttk.Frame(nb)
        nb.add(self.frm_presc, text="Prescriptions")

        self.frm_levels = ttk.Frame(nb)
        nb.add(self.frm_levels, text="%Rx → Gy (auto)")

        self.frm_marg = ttk.Frame(nb)
        nb.add(self.frm_marg, text="Marges")

        self.frm_ring = ttk.Frame(nb)
        nb.add(self.frm_ring, text="Anneaux (rings)")

        self.frm_ruler = ttk.Frame(nb)
        nb.add(self.frm_ruler, text="Ruler gradient")

        self.frm_weights = ttk.Frame(nb)
        nb.add(self.frm_weights, text="Pondération (assistant)")

        self.frm_steps = ttk.Frame(nb)
        nb.add(self.frm_steps, text="Checklist optimisation")

        self.frm_calc = ttk.Frame(nb)
        nb.add(self.frm_calc, text="Calculette Dose ↔ %")

        self.frm_lint = ttk.Frame(nb)
        nb.add(self.frm_lint, text="Nomenclature lint")

        self.frm_dvh = ttk.Frame(nb)
        nb.add(self.frm_dvh, text="DVH & Indices (opt.)")

        # Détails par onglet
        self._build_prescriptions(self.frm_presc)
        self._build_levels(self.frm_levels)
        self._build_margins(self.frm_marg)
        self._build_rings(self.frm_ring)
        self._build_ruler(self.frm_ruler)
        self._build_weights(self.frm_weights)
        self._build_steps(self.frm_steps)
        self._build_calc(self.frm_calc)
        self._build_lint(self.frm_lint)
        self._build_dvh(self.frm_dvh)

    # ----------------------------------------------------- Prescriptions -----
    def _build_prescriptions(self, parent: tk.Widget) -> None:
        """Construit l’onglet de saisie des structures et prescriptions."""
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=6, padx=8)

        ttk.Label(top, text="Gradient par défaut g [%/mm] :").pack(side="left")
        ttk.Entry(top, textvariable=self.gradient_default, width=8).pack(side="left", padx=4)

        ttk.Label(top, text="Préfixe optimisation (TG-263) :").pack(side="left", padx=10)
        self.prefix_combo = ttk.Combobox(
            top,
            values=list(DEFAULT_PREFIX_CHOICES),
            width=4,
            state="readonly",
            textvariable=self.prefix_choice,
        )
        self.prefix_combo.pack(side="left")

        table = ttk.Frame(parent)
        table.pack(fill="both", expand=True, padx=8, pady=8)
        cols = ("Type", "Nom", "Rx [Gy]", "a (gEUD) [—]")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=230 if c == "Nom" else 140, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscroll=vsb.set)

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Ajouter PTV", command=lambda: self._add_struct_row("PTV")).pack(side="left", padx=4)
        ttk.Button(btns, text="Ajouter OAR", command=lambda: self._add_struct_row("OAR")).pack(side="left", padx=4)
        ttk.Button(btns, text="Ajouter PRV", command=lambda: self._add_struct_row("PRV")).pack(side="left", padx=4)
        ttk.Button(btns, text="Supprimer sélection", command=self._del_selected).pack(side="left", padx=12)
        ttk.Button(btns, text="Mettre à jour listes", command=self._refresh_structure_lists).pack(side="left", padx=12)

        ttk.Label(
            parent,
            text=(
                "Rappel TG-263 : préfixer les structures d’optimisation (z/_). "
                "Les structures d’évaluation restent sans préfixe."
            ),
        ).pack(anchor="w", padx=8, pady=4)

    def _add_struct_row(self, structure_type: str) -> None:
        """
        Fenêtre modale pour ajouter une structure.

        :param structure_type: 'PTV', 'OAR' ou 'PRV'.
        """
        win = tk.Toplevel(self)
        win.title(f"Ajouter {structure_type}")
        ttk.Label(win, text="Nom :").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        e_name = ttk.Entry(win, width=28)
        e_name.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(win, text="Rx [Gy] (pour PTV) :").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        e_rx = ttk.Entry(win, width=10)
        e_rx.grid(row=1, column=1, padx=4, pady=4)

        ttk.Label(win, text="a (gEUD) [—] :").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        e_a = ttk.Entry(win, width=10)
        e_a.grid(row=2, column=1, padx=4, pady=4)
        e_a.insert(0, "-10" if structure_type == "PTV" else "10")

        def add_ok() -> None:
            """Valide la saisie et ajoute la structure à la table."""
            name = e_name.get().strip()
            if not name:
                messagebox.showerror("Erreur", "Nom requis.")
                return
            rx = float(e_rx.get()) if e_rx.get().strip() else None
            a = float(e_a.get()) if e_a.get().strip() else None
            self.structures.append(Structure(name=name, structure_type=structure_type, rx=rx, a_param=a))
            self.tree.insert(
                "",
                "end",
                values=(structure_type, name, "" if rx is None else format_value(rx), "" if a is None else format_value(a, 0)),
            )
            win.destroy()
            self._refresh_structure_lists()

        ttk.Button(win, text="Ajouter", command=add_ok).grid(row=3, column=0, columnspan=2, pady=6)

    def _del_selected(self) -> None:
        """Supprime les structures sélectionnées dans la table."""
        for sel in self.tree.selection():
            vals = self.tree.item(sel, "values")
            self.structures = [
                s for s in self.structures if not (s.structure_type == vals[0] and s.name == vals[1])
            ]
            self.tree.delete(sel)
        self._refresh_structure_lists()

    def _refresh_structure_lists(self) -> None:
        """Met à jour les listes déroulantes dépendantes des structures."""
        ptv_names = [s.name for s in self.structures if s.structure_type == "PTV"]
        all_names = [s.name for s in self.structures]
        if hasattr(self, "ptv_combo"):
            self.ptv_combo["values"] = ptv_names
        if hasattr(self, "mg_struct_a_combo"):
            self.mg_struct_a_combo["values"] = all_names
        if hasattr(self, "mg_struct_b_combo"):
            self.mg_struct_b_combo["values"] = all_names
        if hasattr(self, "calc_ptv_combo"):
            self.calc_ptv_combo["values"] = ptv_names

    # ---------------------------------------------- %Rx → Gy (sans DVH) -----
    def _build_levels(self, parent: tk.Widget) -> None:
        """Construit l’onglet calculant les niveaux %Rx en Gy pour les PTV."""
        ttk.Label(parent, text="Niveaux en Gy calculés directement à partir de Rx (sans DVH).").pack(
            anchor="w", padx=8, pady=6
        )
        cols = ("PTV", "Rx [Gy]", "80% [Gy]", "90% [Gy]", "95% [Gy]", "98% [Gy]", "100% [Gy]", "105% [Gy]", "107% [Gy]", "110% [Gy]")
        self.levels_tree = ttk.Treeview(parent, columns=cols, show="headings", height=16)
        for c in cols:
            self.levels_tree.heading(c, text=c)
            w = 170 if c == "PTV" else 110
            self.levels_tree.column(c, width=w, anchor="center")
        self.levels_tree.pack(fill="both", expand=True, padx=8, pady=8)
        ttk.Button(parent, text="Recalculer", command=self._recalc_levels).pack(anchor="w", padx=8, pady=4)

    def _recalc_levels(self) -> None:
        """Recalcule et affiche les niveaux en Gy pour chaque PTV avec Rx valide."""
        self.levels_tree.delete(*self.levels_tree.get_children())
        for s in self.structures:
            if s.structure_type == "PTV" and s.rx and s.rx > 0:
                row = [s.name, format_value(s.rx)]
                row += [format_value(s.rx * p) for p in DEFAULT_PCT_LEVELS]
                self.levels_tree.insert("", "end", values=row)

    # ----------------------------------------------------------- Marges -----
    def _build_margins(self, parent: tk.Widget) -> None:
        """Construit l’onglet d’estimation de marge via g [%/mm]."""
        modef = ttk.LabelFrame(parent, text="Mode de saisie")
        modef.pack(fill="x", padx=8, pady=8)
        self.margin_mode = tk.StringVar(value="structures")
        ttk.Radiobutton(
            modef,
            text="Depuis structures (Rx×% ou absolue)",
            variable=self.margin_mode,
            value="structures",
            command=self._show_margin_mode,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            modef, text="Manuel (D1/D2 en Gy)", variable=self.margin_mode, value="manuel", command=self._show_margin_mode
        ).pack(side="left", padx=8)

        out = ttk.Frame(parent)
        out.pack(fill="x", padx=8, pady=(0, 8))
        self.m_delta = tk.StringVar(value="Δ [%] = —")
        self.m_marg = tk.StringVar(value="m [mm] = —")
        ttk.Label(out, textvariable=self.m_delta).pack(side="left", padx=8)
        ttk.Label(out, textvariable=self.m_marg).pack(side="left", padx=12)

        self.bridge_lbl = tk.StringVar(value="Suggestion anneau (transition) : —")
        ttk.Label(parent, textvariable=self.bridge_lbl).pack(fill="x", padx=8, pady=(0, 8))

        # ---- Manuel
        self.m_manual = ttk.LabelFrame(parent, text="Marge (mode MANUEL)")
        self.m_d1 = tk.DoubleVar(value=70.0)
        self.m_d2 = tk.DoubleVar(value=56.0)
        self.m_g = tk.DoubleVar(value=float(self.gradient_default.get()))
        ttk.Label(self.m_manual, text="D1 [Gy] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(self.m_manual, textvariable=self.m_d1, width=10).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(self.m_manual, text="D2 [Gy] :").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        ttk.Entry(self.m_manual, textvariable=self.m_d2, width=10).grid(row=0, column=3, padx=4, pady=4)
        ttk.Label(self.m_manual, text="g [%/mm] :").grid(row=0, column=4, padx=4, pady=4, sticky="e")
        ttk.Entry(self.m_manual, textvariable=self.m_g, width=10).grid(row=0, column=5, padx=4, pady=4)
        ttk.Button(self.m_manual, text="Calculer", command=self._calc_grad_margin).grid(row=0, column=6, padx=6)
        ttk.Button(
            self.m_manual,
            text="g ← défaut",
            command=lambda: self.m_g.set(float(self.gradient_default.get())),
        ).grid(row=0, column=7, padx=6)

        # ---- Structures
        self.m_struct = ttk.LabelFrame(parent, text="Marge (mode STRUCTURES)")
        self.mg_g = tk.DoubleVar(value=float(self.gradient_default.get()))
        ttk.Label(self.m_struct, text="g [%/mm] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(self.m_struct, textvariable=self.mg_g, width=10).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(
            self.m_struct, text="g ← défaut", command=lambda: self.mg_g.set(float(self.gradient_default.get()))
        ).grid(row=0, column=2, padx=6)

        pct_values = ["110", "107", "105", "100", "98", "95", "90", "85", "80", "70", "60"]
        all_names = [s.name for s in self.structures]

        ttk.Label(self.m_struct, text="Structure A :").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        self.mg_struct_a_combo = ttk.Combobox(self.m_struct, values=all_names, width=24, state="readonly")
        self.mg_struct_a_combo.grid(row=1, column=1, padx=4, pady=4)
        self.mg_src_a = tk.StringVar(value="rxpct")
        ttk.Radiobutton(self.m_struct, text="Rx×% :", variable=self.mg_src_a, value="rxpct", command=self._on_src_change_a).grid(
            row=1, column=2, padx=6, pady=4, sticky="e"
        )
        ttk.Radiobutton(self.m_struct, text="Absolue [Gy] :", variable=self.mg_src_a, value="abs", command=self._on_src_change_a).grid(
            row=1, column=3, padx=6, pady=4, sticky="e"
        )
        self.mg_pct_a = ttk.Combobox(self.m_struct, values=pct_values, width=6, state="readonly")
        self.mg_pct_a.set("100")
        self.mg_pct_a.grid(row=1, column=4, padx=4, pady=4)
        self.mg_abs_a = tk.DoubleVar(value=0.0)
        self.mg_abs_a_ent = ttk.Entry(self.m_struct, textvariable=self.mg_abs_a, width=10, state="disabled")
        self.mg_abs_a_ent.grid(row=1, column=5, padx=4, pady=4)

        ttk.Label(self.m_struct, text="Structure B :").grid(row=2, column=0, padx=4, pady=4, sticky="e")
        self.mg_struct_b_combo = ttk.Combobox(self.m_struct, values=all_names, width=24, state="readonly")
        self.mg_struct_b_combo.grid(row=2, column=1, padx=4, pady=4)
        self.mg_src_b = tk.StringVar(value="rxpct")
        ttk.Radiobutton(self.m_struct, text="Rx×% :", variable=self.mg_src_b, value="rxpct", command=self._on_src_change_b).grid(
            row=2, column=2, padx=6, pady=4, sticky="e"
        )
        ttk.Radiobutton(self.m_struct, text="Absolue [Gy] :", variable=self.mg_src_b, value="abs", command=self._on_src_change_b).grid(
            row=2, column=3, padx=6, pady=4, sticky="e"
        )
        self.mg_pct_b = ttk.Combobox(self.m_struct, values=pct_values, width=6, state="readonly")
        self.mg_pct_b.set("100")
        self.mg_pct_b.grid(row=2, column=4, padx=4, pady=4)
        self.mg_abs_b = tk.DoubleVar(value=0.0)
        self.mg_abs_b_ent = ttk.Entry(self.m_struct, textvariable=self.mg_abs_b, width=10, state="disabled")
        self.mg_abs_b_ent.grid(row=2, column=5, padx=4, pady=4)

        ttk.Button(self.m_struct, text="Calculer marge (structures)", command=self._calc_margin_from_structs).grid(
            row=3, column=0, columnspan=6, padx=6, pady=8
        )

        self._show_margin_mode()

    def _show_margin_mode(self) -> None:
        """Affiche le sous-panneau correspondant au mode de marge choisi."""
        try:
            self.m_manual.pack_forget()
            self.m_struct.pack_forget()
        except Exception:
            pass
        if self.margin_mode.get() == "manuel":
            self.m_manual.pack(fill="x", padx=8, pady=8)
        else:
            self.m_struct.pack(fill="x", padx=8, pady=8)

    def _on_src_change_a(self) -> None:
        """Active/désactive les entrées pour la Structure A selon la source."""
        if self.mg_src_a.get() == "rxpct":
            self.mg_pct_a.configure(state="readonly")
            self.mg_abs_a_ent.configure(state="disabled")
        else:
            self.mg_pct_a.configure(state="disabled")
            self.mg_abs_a_ent.configure(state="normal")

    def _on_src_change_b(self) -> None:
        """Active/désactive les entrées pour la Structure B selon la source."""
        if self.mg_src_b.get() == "rxpct":
            self.mg_pct_b.configure(state="readonly")
            self.mg_abs_b_ent.configure(state="disabled")
        else:
            self.mg_pct_b.configure(state="disabled")
            self.mg_abs_b_ent.configure(state="normal")

    def _get_structure_by_name(self, name: str) -> Optional[Structure]:
        """
        Renvoie l’objet Structure correspondant au nom.

        :param name: Nom de structure exact.
        :return: Structure ou None.
        """
        for s in self.structures:
            if s.name == name:
                return s
        return None

    def _update_bridge_suggestion(self, d1: float, d2: float, g: float) -> None:
        """
        Met à jour la suggestion d’anneau de transition basée sur m = Δ/g.

        :param d1: Dose haute [Gy].
        :param d2: Dose basse [Gy].
        :param g: Gradient [%/mm].
        """
        if d1 <= 0 or d2 <= 0 or g <= 0:
            self.bridge_lbl.set("Suggestion anneau (transition) : —")
            return
        delta_pct = 100.0 * (1.0 - d2 / d1)
        mmm = delta_pct / g  # [mm]
        dmax_pct = max(0.0, 100.0 - g * (0.0 + float(self.boundary_delta.get())))
        dmean_pct = max(0.0, 100.0 - g * (0.5 * mmm))
        self.bridge_lbl.set(
            f"Suggestion : anneau 0–{mmm:.1f} mm → Dmax ≤ {dmax_pct:.1f} %Rx ; Dmean ≤ {dmean_pct:.1f} %Rx"
        )

    def _calc_margin_from_structs(self) -> None:
        """Calcule la marge m [mm] à partir de deux structures et de g."""
        a_name = self.mg_struct_a_combo.get().strip()
        b_name = self.mg_struct_b_combo.get().strip()
        if not a_name or not b_name:
            messagebox.showerror("Erreur", "Choisissez les deux structures (A et B).")
            return
        s_a = self._get_structure_by_name(a_name)
        s_b = self._get_structure_by_name(b_name)
        if s_a is None or s_b is None:
            messagebox.showerror("Erreur", "Structure introuvable.")
            return

        # D1
        if self.mg_src_a.get() == "rxpct":
            if not s_a.rx or s_a.rx <= 0:
                messagebox.showerror("Erreur", f"'{a_name}' n'a pas de Rx définie.")
                return
            p_a = float(self.mg_pct_a.get())
            d1 = s_a.rx * p_a / 100.0
        else:
            d1 = float(self.mg_abs_a.get())

        # D2
        if self.mg_src_b.get() == "rxpct":
            if not s_b.rx or s_b.rx <= 0:
                messagebox.showerror("Erreur", f"'{b_name}' n'a pas de Rx définie.")
                return
            p_b = float(self.mg_pct_b.get())
            d2 = s_b.rx * p_b / 100.0
        else:
            d2 = float(self.mg_abs_b.get())

        g = float(self.mg_g.get())
        if d1 <= 0 or d2 <= 0 or g <= 0:
            messagebox.showerror("Erreur", "D1, D2 et g doivent être > 0.")
            return
        if d2 > d1:
            d1, d2 = d2, d1

        delta = 100.0 * (1.0 - d2 / d1)  # [%]
        m = delta / g  # [mm]
        self.m_delta.set(f"Δ [%] = {delta:.2f}")
        self.m_marg.set(f"m [mm] = {m:.2f}")
        self._update_bridge_suggestion(d1, d2, g)

    def _calc_grad_margin(self) -> None:
        """Calcule la marge m [mm] à partir de D1/D2 [Gy] et g [%/mm] (mode manuel)."""
        d1 = float(self.m_d1.get())
        d2 = float(self.m_d2.get())
        g = float(self.m_g.get())
        if d1 <= 0 or d2 <= 0 or g <= 0:
            messagebox.showerror("Erreur", "D1, D2 et g doivent être > 0.")
            return
        if d2 > d1:
            d1, d2 = d2, d1
        delta = 100.0 * (1.0 - d2 / d1)
        m = delta / g
        self.m_delta.set(f"Δ [%] = {delta:.2f}")
        self.m_marg.set(f"m [mm] = {m:.2f}")
        self._update_bridge_suggestion(d1, d2, g)

    # ------------------------------------------------------------ Rings -----
    def _build_rings(self, parent: tk.Widget) -> None:
        """Construit l’onglet de génération d’anneaux et contraintes associées."""
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="PTV :").pack(side="left")
        self.ptv_combo = ttk.Combobox(top, values=[], width=24, state="readonly")
        self.ptv_combo.pack(side="left", padx=6)

        ttk.Label(top, text="Préfixe opti :").pack(side="left", padx=6)
        ttk.Combobox(top, values=list(DEFAULT_PREFIX_CHOICES), width=4, state="readonly", textvariable=self.prefix_choice).pack(
            side="left"
        )

        ttk.Label(top, text="g [%/mm] :").pack(side="left", padx=6)
        self.r_g = tk.DoubleVar(value=DEFAULT_GRADIENT)
        ttk.Entry(top, textvariable=self.r_g, width=8).pack(side="left")
        ttk.Button(top, text="g ← défaut", command=lambda: self.r_g.set(float(self.gradient_default.get()))).pack(
            side="left", padx=6
        )

        ttk.Label(top, text="δ [mm] pour Dmax :").pack(side="left", padx=6)
        ttk.Entry(top, textvariable=self.boundary_delta, width=6).pack(side="left")

        ttk.Label(top, text="Anneaux (mm) ex. 0-5, 5-10, 10-20 :").pack(side="left", padx=6)
        self.r_entries: List[ttk.Entry] = []
        for default in RING_DEFAULTS_MM:
            e = ttk.Entry(top, width=10)
            e.insert(0, default)
            e.pack(side="left", padx=3)
            self.r_entries.append(e)

        ttk.Button(top, text="Ajouter rings", command=self._add_rings_for_ptv).pack(side="left", padx=8)
        ttk.Button(top, text="Supprimer sélection", command=self._del_selected_rings).pack(side="left", padx=6)
        ttk.Button(top, text="Vider table", command=self._clear_rings).pack(side="left", padx=6)

        rf = ttk.Frame(parent)
        rf.pack(fill="both", expand=True, padx=8, pady=8)
        cols = (
            "PTV",
            "ROI proposé",
            "r_in [mm]",
            "r_out [mm]",
            "épaisseur [mm]",
            "Dmax [%Rx]",
            "Dmax [Gy]",
            "Dmean [%Rx]",
            "Dmean [Gy]",
        )
        self.ring_tree = ttk.Treeview(rf, columns=cols, show="headings", height=18)
        for c in cols:
            self.ring_tree.heading(c, text=c)
            self.ring_tree.column(c, width=170 if c == "ROI proposé" else 120, anchor="center")
        self.ring_tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(rf, orient="vertical", command=self.ring_tree.yview)
        vsb.pack(side="right", fill="y")
        self.ring_tree.configure(yscroll=vsb.set)

        ttk.Label(
            parent,
            text=(
                "Formules : Dmax(r_in) ≤ Rx·(1 − g·(r_in+δ)/100),  "
                "Dmean(centre) ≤ Rx·(1 − g·(r_in+w/2)/100)."
            ),
        ).pack(anchor="w", padx=8, pady=4)

    def _make_ring_name(self, ptv_name: str, rin: float, rout: float) -> str:
        """
        Crée un nom d’anneau compatible TG-263 avec préfixe.

        :param ptv_name: Nom du PTV.
        :param rin: Rayon interne [mm].
        :param rout: Rayon externe [mm].
        :return: Nom ROI proposé.
        """
        pre = self.prefix_choice.get() or DEFAULT_PREFIX_CHOICES[0]
        return f"{pre}Ring_{ptv_name}_{pad_mm(rin)}_{pad_mm(rout)}"

    def _add_rings_for_ptv(self) -> None:
        """Ajoute les lignes d’anneaux calculées pour le PTV sélectionné."""
        name = self.ptv_combo.get().strip()
        if not name:
            messagebox.showerror("Erreur", "Choisissez un PTV.")
            return
        ptv = next((s for s in self.structures if s.structure_type == "PTV" and s.name == name), None)
        if (ptv is None) or (ptv.rx is None) or (ptv.rx <= 0):
            messagebox.showerror("Erreur", "PTV introuvable ou Rx invalide.")
            return

        g = self.r_g.get()
        delta = max(0.0, self.boundary_delta.get())
        if g <= 0:
            messagebox.showerror("Erreur", "g doit être > 0.")
            return

        ranges: List[Tuple[float, float]] = []
        for e in self.r_entries:
            txt = e.get().strip().replace(" ", "")
            if not txt:
                continue
            try:
                rin_s, rout_s = txt.split("-")
                rin, rout = float(rin_s), float(rout_s)
                if rout <= rin:
                    raise ValueError("r_out doit être > r_in")
                ranges.append((rin, rout))
            except Exception as ex:
                messagebox.showerror("Erreur", f"Intervalle invalide: '{txt}'. Format: a-b (mm). Détail: {ex}")
                return

        for rin, rout in ranges:
            w = rout - rin
            dmax_pct = max(0.0, 1.0 - g * (rin + delta) / 100.0) * 100.0
            dmean_pct = max(0.0, 1.0 - g * (rin + 0.5 * w) / 100.0) * 100.0
            dmax_gy = ptv.rx * dmax_pct / 100.0
            dmean_gy = ptv.rx * dmean_pct / 100.0
            roi_name = self._make_ring_name(ptv.name, rin, rout)
            self.ring_tree.insert(
                "",
                "end",
                values=(
                    ptv.name,
                    roi_name,
                    format_value(rin, 1),
                    format_value(rout, 1),
                    format_value(w, 1),
                    format_value(dmax_pct, 1),
                    format_value(dmax_gy, 2),
                    format_value(dmean_pct, 1),
                    format_value(dmean_gy, 2),
                ),
            )

    def _del_selected_rings(self) -> None:
        """Supprime les anneaux sélectionnés dans la table."""
        for sel in self.ring_tree.selection():
            self.ring_tree.delete(sel)

    def _clear_rings(self) -> None:
        """Vide la table des anneaux."""
        self.ring_tree.delete(*self.ring_tree.get_children())

    # ------------------------------------------------------ Ruler Gradient ---
    def _build_ruler(self, parent: tk.Widget) -> None:
        """Construit l’onglet de conversions avec le gradient g [%/mm]."""
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(parent, text="Convention : r [mm] mesuré depuis la surface PTV (100 % Rx au bord).").pack(
            anchor="w", padx=8
        )

        self.rul_g = tk.DoubleVar(value=float(self.gradient_default.get()))
        ttk.Label(top, text="g [%/mm] :").pack(side="left")
        ttk.Entry(top, textvariable=self.rul_g, width=8).pack(side="left", padx=4)
        ttk.Button(top, text="g ← défaut", command=lambda: self.rul_g.set(float(self.gradient_default.get()))).pack(
            side="left", padx=6
        )

        b1 = ttk.LabelFrame(parent, text="Distance pour passer de P1 à P2")
        b1.pack(fill="x", padx=8, pady=8)
        self.rul_p1 = tk.DoubleVar(value=100.0)
        self.rul_p2 = tk.DoubleVar(value=90.0)
        ttk.Label(b1, text="P1 [%] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(b1, textvariable=self.rul_p1, width=10).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(b1, text="P2 [%] :").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        ttk.Entry(b1, textvariable=self.rul_p2, width=10).grid(row=0, column=3, padx=4, pady=4)
        self.rul_m_out = tk.StringVar(value="m [mm] = —")
        ttk.Button(b1, text="Calculer", command=self._rul_p_to_mm).grid(row=0, column=4, padx=6)
        ttk.Label(b1, textvariable=self.rul_m_out).grid(row=0, column=5, padx=6)

        b2 = ttk.LabelFrame(parent, text="Pour une distance r depuis 100 %")
        b2.pack(fill="x", padx=8, pady=8)
        self.rul_r = tk.DoubleVar(value=10.0)
        ttk.Label(b2, text="r [mm] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(b2, textvariable=self.rul_r, width=10).grid(row=0, column=1, padx=4, pady=4)
        self.rul_p_out = tk.StringVar(value="P(r) [%] ≈ —")
        ttk.Button(b2, text="Calculer", command=self._rul_mm_to_p).grid(row=0, column=2, padx=6)
        ttk.Label(b2, textvariable=self.rul_p_out).grid(row=0, column=3, padx=6)

        b3 = ttk.LabelFrame(parent, text="Conversion % ↔ Gy pour une Rx")
        b3.pack(fill="x", padx=8, pady=8)
        self.rul_rx = tk.DoubleVar(value=70.0)
        self.rul_pct = tk.DoubleVar(value=95.0)
        self.rul_dose = tk.DoubleVar(value=0.0)
        ttk.Label(b3, text="Rx [Gy] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(b3, textvariable=self.rul_rx, width=10).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(b3, text="%Rx [%] :").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        ttk.Entry(b3, textvariable=self.rul_pct, width=10).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(b3, text="% → Gy", command=self._rul_pct_to_gy).grid(row=0, column=4, padx=6)
        ttk.Label(b3, text="Dose [Gy] :").grid(row=0, column=5, padx=4, pady=4)
        ttk.Entry(b3, textvariable=self.rul_dose, width=10).grid(row=0, column=6, padx=4, pady=4)
        ttk.Button(b3, text="Gy → %", command=self._rul_gy_to_pct).grid(row=0, column=7, padx=6)

    def _rul_p_to_mm(self) -> None:
        """Calcule la distance m [mm] pour passer de P1 à P2 via g [%/mm]."""
        g = float(self.rul_g.get())
        p1 = float(self.rul_p1.get())
        p2 = float(self.rul_p2.get())
        if g <= 0:
            self.rul_m_out.set("m [mm] = —")
            return
        m = abs(p1 - p2) / g
        self.rul_m_out.set(f"m [mm] = {m:.2f}")

    def _rul_mm_to_p(self) -> None:
        """Estime P(r) [%] à une distance r [mm] depuis 100 %Rx."""
        g = float(self.rul_g.get())
        r = float(self.rul_r.get())
        p = max(0.0, 100.0 - g * max(0.0, r))
        self.rul_p_out.set(f"P(r) [%] ≈ {p:.1f}")

    def _rul_pct_to_gy(self) -> None:
        """Convertit %Rx → Gy pour la Rx choisie."""
        rx = float(self.rul_rx.get())
        pct = float(self.rul_pct.get())
        self.rul_dose.set(rx * pct / 100.0)

    def _rul_gy_to_pct(self) -> None:
        """Convertit Gy → %Rx pour la Rx choisie."""
        rx = float(self.rul_rx.get())
        dose = float(self.rul_dose.get())
        self.rul_pct.set(100.0 * dose / rx if rx > 0 else 0.0)

    # -------------------------------------------- Conseiller pondérations ---
    def _build_weights(self, parent: tk.Widget) -> None:
        """Construit l’onglet d’heuristique de pondération des objectifs."""
        ttk.Label(parent, text="Choisir les priorités → ratio de poids suggéré (heuristique).").pack(
            anchor="w", padx=8, pady=6
        )
        grid = ttk.Frame(parent)
        grid.pack(fill="x", padx=8, pady=8)
        self.w_levels: Dict[str, ttk.Combobox] = {}
        items = [("Couverture PTV", "PTV"), ("Retombée / Rings", "RING"), ("OARs proches", "OAR"), ("Lissage / régularité", "SMOOTH")]
        choices = ["Bas", "Moyen", "Haut"]
        colnames = ["Objectif", "Priorité", "Poids relatif [—]"]
        for j, name in enumerate(colnames):
            ttk.Label(grid, text=name).grid(row=0, column=j, padx=6, pady=4, sticky="w")
        self.w_weight_vars: Dict[str, tk.StringVar] = {}
        for i, (label, key) in enumerate(items, start=1):
            ttk.Label(grid, text=label).grid(row=i, column=0, padx=6, pady=4, sticky="w")
            cb = ttk.Combobox(grid, values=choices, state="readonly", width=8)
            cb.set("Haut" if key in ("PTV", "OAR") else "Moyen")
            cb.grid(row=i, column=1, padx=6, pady=4)
            self.w_levels[key] = cb
            v = tk.StringVar(value="—")
            ttk.Label(grid, textvariable=v).grid(row=i, column=2, padx=6, pady=4, sticky="w")
            self.w_weight_vars[key] = v
        ttk.Button(parent, text="Suggérer ratios", command=self._compute_weights).pack(anchor="w", padx=8, pady=6)
        self.w_out = tk.StringVar(value="Ratios → —")
        ttk.Label(parent, textvariable=self.w_out).pack(anchor="w", padx=8, pady=6)
        self.w_hint = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.w_hint, wraplength=1200, justify="left").pack(fill="x", padx=8, pady=(0, 10))

    def _compute_weights(self) -> None:
        """Calcule des ratios de poids relatifs à partir des priorités choisies."""
        factor = {"Bas": 0.15, "Moyen": 0.4, "Haut": 1.0}
        base = 100.0
        ptv = factor[self.w_levels["PTV"].get()]
        ring = factor[self.w_levels["RING"].get()]
        oar = factor[self.w_levels["OAR"].get()]
        smo = factor[self.w_levels["SMOOTH"].get()]
        if ptv == 0:
            ptv = 0.01
        s = base / ptv
        w_ptv = int(round(base))
        w_ring = int(round(ring * s * base / 100.0))
        w_oar = int(round(oar * s * base / 100.0))
        w_smo = int(round(smo * s * base / 100.0))
        self.w_weight_vars["PTV"].set(str(w_ptv))
        self.w_weight_vars["RING"].set(str(w_ring))
        self.w_weight_vars["OAR"].set(str(w_oar))
        self.w_weight_vars["SMOOTH"].set(str(w_smo))
        self.w_out.set(f"Ratios suggérés : PTV:{w_ptv} / Rings:{w_ring} / OAR:{w_oar} / Lissage:{w_smo}")
        hint = []
        if self.w_levels["PTV"].get() == "Haut" and self.w_levels["OAR"].get() == "Haut":
            hint.append("Astuce : optimisation en étapes (PTV puis OAR) ou objectifs durs/élastiques différenciés.")
        if self.w_levels["RING"].get() == "Haut":
            hint.append("Si hotspots > 107 % Rx, augmenter poids anneau 0–5 mm ; décroître avec la distance.")
        if self.w_levels["SMOOTH"].get() == "Haut":
            hint.append("Le lissage peut baisser la couverture : revérifier D95 %.")
        self.w_hint.set(" ".join(hint))

    # ----------------------------------------------- Checklist / étapes -----
    def _build_steps(self, parent: tk.Widget) -> None:
        """Construit l’onglet de checklist d’optimisation."""
        self.steps_vars: List[tk.BooleanVar] = []
        ttk.Label(parent, text=lambda: None)  # placeholder
        self._render_steps(parent)

    def _render_steps(self, parent: tk.Widget) -> None:
        """Rafraîchit le contenu de la checklist avec la valeur de g actuelle."""
        for child in parent.winfo_children():
            child.destroy()
        gtxt = f"{float(self.gradient_default.get()):.1f}"
        ttk.Label(parent, text=f"Séquence type (adapter selon le site) – g ACTUEL = {gtxt} %/mm :").pack(
            anchor="w", padx=8, pady=6
        )
        steps = [
            (
                "Étape 1 – Couverture PTV & homogénéité",
                "Viser D95% ≥ 95–98 %Rx ; D2% ≤ 107 %Rx ; Dmax ≤ 110 %Rx ; poser anneau 0–5 mm.",
            ),
            (
                "Étape 2 – Retombée (rings) & confinement dose",
                "Dmax/Dmean pour 0–5, 5–10, 10–20 mm, en cohérence avec g ACTUEL ; décroître les poids avec la distance.",
            ),
            (
                "Étape 3 – OARs proches / PRV",
                "Prioriser les organes limitants ; régler les anneaux si conflit ; vérifier PRV si besoin (van Herk).",
            ),
            ("Étape 4 – Polish & lissage", "Réduire oscillations ; s’assurer qu’aucun objectif critique ne régresse."),
        ]
        self.steps_vars = []
        for title, desc in steps:
            var = tk.BooleanVar(value=False)
            frm = ttk.Frame(parent)
            frm.pack(fill="x", padx=8, pady=4)
            ttk.Checkbutton(frm, text=title, variable=var, command=self._update_next_step_hint).pack(anchor="w")
            ttk.Label(frm, text=desc, foreground="#555").pack(anchor="w", padx=24)
            self.steps_vars.append(var)
        self.next_hint = tk.StringVar(value="Prochaine action : Étape 1.")
        ttk.Label(parent, textvariable=self.next_hint, foreground="#006", wraplength=1200, justify="left").pack(
            fill="x", padx=8, pady=8
        )

    def _update_next_step_hint(self) -> None:
        """Met à jour la recommandation d’étape suivante selon les coches."""
        done = [v.get() for v in self.steps_vars]
        if not done[0]:
            self.next_hint.set("Prochaine action : Étape 1 – poser objectifs PTV + anneau 0–5 mm.")
        elif not done[1]:
            self.next_hint.set("Prochaine action : Étape 2 – retombée (anneaux) avec g ACTUEL.")
        elif not done[2]:
            self.next_hint.set("Prochaine action : Étape 3 – OAR/PRV prioritaires.")
        elif not done[3]:
            self.next_hint.set("Prochaine action : Étape 4 – polish & lissage.")
        else:
            self.next_hint.set("Checklist complétée ✅")

    # ------------------------------------------------------ Calculette -----
    def _build_calc(self, parent: tk.Widget) -> None:
        """Construit l’onglet de calculette Dose ↔ %Rx."""
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="PTV (pour Rx auto) :").pack(side="left")
        self.calc_ptv_combo = ttk.Combobox(top, values=[], width=24, state="readonly")
        self.calc_ptv_combo.pack(side="left", padx=6)
        ttk.Button(top, text="Utiliser Rx du PTV", command=self._calc_use_ptv_rx).pack(side="left", padx=6)

        mid = ttk.Frame(parent)
        mid.pack(fill="x", padx=8, pady=8)
        self.calc_rx = tk.DoubleVar(value=70.0)
        ttk.Label(mid, text="Rx [Gy] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(mid, textvariable=self.calc_rx, width=10).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(mid, text="%Rx [%] → Dose [Gy] :").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        self.calc_pct = tk.DoubleVar(value=95.0)
        ttk.Entry(mid, textvariable=self.calc_pct, width=10).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(mid, text="Calculer", command=self._calc_pct_to_gy).grid(row=1, column=2, padx=6)
        self.calc_gy_out = tk.StringVar(value="Dose [Gy] = —")
        ttk.Label(mid, textvariable=self.calc_gy_out).grid(row=1, column=3, padx=6, pady=4, sticky="w")

        ttk.Label(mid, text="Dose [Gy] → %Rx [%] :").grid(row=2, column=0, padx=4, pady=4, sticky="e")
        self.calc_gy_in = tk.DoubleVar(value=66.5)
        ttk.Entry(mid, textvariable=self.calc_gy_in, width=10).grid(row=2, column=1, padx=4, pady=4)
        ttk.Button(mid, text="Calculer", command=self._calc_gy_to_pct).grid(row=2, column=2, padx=6)
        self.calc_pct_out = tk.StringVar(value="%Rx [%] = —")
        ttk.Label(mid, textvariable=self.calc_pct_out).grid(row=2, column=3, padx=6, pady=4, sticky="w")

    def _calc_use_ptv_rx(self) -> None:
        """Récupère la Rx du PTV sélectionné pour la calculette."""
        name = self.calc_ptv_combo.get().strip()
        if not name:
            messagebox.showerror("Erreur", "Choisissez un PTV.")
            return
        ptv = next((s for s in self.structures if s.structure_type == "PTV" and s.name == name), None)
        if ptv is None or not ptv.rx or ptv.rx <= 0:
            messagebox.showerror("Erreur", "PTV introuvable ou Rx non définie.")
            return
        self.calc_rx.set(ptv.rx)

    def _calc_pct_to_gy(self) -> None:
        """Calcule Gy à partir de %Rx pour la Rx indiquée."""
        rx = float(self.calc_rx.get())
        pct = float(self.calc_pct.get())
        self.calc_gy_out.set(f"Dose [Gy] = {rx * pct / 100.0:.3f}")

    def _calc_gy_to_pct(self) -> None:
        """Calcule %Rx à partir d’une dose Gy pour la Rx indiquée."""
        rx = float(self.calc_rx.get())
        dval = float(self.calc_gy_in.get())
        self.calc_pct_out.set(f"%Rx [%] = {100.0 * dval / rx:.2f}" if rx > 0 else "%Rx [%] = —")

    # -------------------------------------------------------- Nomenclature ---
    def _build_lint(self, parent: tk.Widget) -> None:
        """Construit l’onglet d’audit de nomenclature (TG-263)."""
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Style PTV pour lint :").pack(side="left")
        ttk.Radiobutton(top, text="Relatif (PTV_High/Mid/Low)", variable=self.ptv_style, value="Relatif").pack(
            side="left", padx=6
        )
        ttk.Radiobutton(top, text="cGy (PTV_7000)", variable=self.ptv_style, value="cGy").pack(side="left", padx=6)
        ttk.Button(top, text="Analyser", command=self._run_lint).pack(side="left", padx=12)

        cols = ("Structure", "Problème", "Suggestion")
        self.lint_tree = ttk.Treeview(parent, columns=cols, show="headings", height=18)
        for c in cols:
            self.lint_tree.heading(c, text=c)
            self.lint_tree.column(c, width=220 if c == "Structure" else 520, anchor="w")
        self.lint_tree.pack(fill="both", expand=True, padx=8, pady=8)

        ttk.Label(
            parent,
            text=(
                "Règles : caractères ASCII alphanumériques + '_' ; pas d’accents/espace/tiret ;\n"
                "structures d’optimisation préfixées (z/_), structures d’évaluation sans préfixe ;\n"
                "PRV avec marge en mm à 2 chiffres (p.ex. SpinalCord_PRV05) ; latéralité `_L`/`_R`.\n"
                "Style PTV (Relatif/cGy) appliqué uniquement comme recommandation de nommage."
            ),
        ).pack(anchor="w", padx=8, pady=4)

    def _run_lint(self) -> None:
        """Exécute des heuristiques simples de lint sur les noms de structures."""
        self.lint_tree.delete(*self.lint_tree.get_children())
        lateral_oars = {
            "Parotid",
            "SubmandibularGland",
            "LacrimalGland",
            "Eye",
            "Lens",
            "OpticNerve",
            "Cochlea",
            "TemporalLobe",
            "Hip",
            "Kidney",
            "Lung",
        }
        prefix_ops = DEFAULT_PREFIX_CHOICES

        # Tri des PTV par Rx pour le style "Relatif"
        ptvs = [s for s in self.structures if s.structure_type == "PTV" and s.rx]
        ptvs_sorted = sorted(ptvs, key=lambda s: (-s.rx, s.name))
        ptv_rank = {s.name: i for i, s in enumerate(ptvs_sorted)}

        for s in self.structures:
            name = s.name
            base_norm = strip_accents_keep_ascii(name)
            if base_norm != name or re.search(r"[^A-Za-z0-9_]", name):
                sug = strip_accents_keep_ascii(name)
                self.lint_tree.insert("", "end", values=(name, "Caractère non conforme (accent/espace/tiret/…)", f"→ {sug}"))

            # Éval/opti : préfixe
            is_prefixed = name.startswith(prefix_ops)
            if s.structure_type in ("PTV", "OAR", "PRV"):
                # Structures d’évaluation ne devraient pas être préfixées
                if is_prefixed and not (re.search(r"ring", name, re.I) or re.search(r"opt", name, re.I)):
                    self.lint_tree.insert("", "end", values=(name, "Structure d’évaluation avec préfixe d’opti", "→ retirer le préfixe z/_"))

            # Rings sans préfixe
            if (re.search(r"ring", name, re.I)) and (not is_prefixed):
                self.lint_tree.insert("", "end", values=(name, "Anneau sans préfixe d’opti", f"→ {self.prefix_choice.get()}" + name))

            # PRV mm 2 chiffres
            m = re.search(r"_PRV(\d+)$", name)
            if m:
                mm = m.group(1)
                if len(mm) != 2:
                    sug = re.sub(r"_PRV(\d+)$", f"_PRV{int(mm):02d}", name)
                    self.lint_tree.insert("", "end", values=(name, "PRV mm non formatté à 2 chiffres", f"→ {sug}"))

            # Latéralité
            for token in lateral_oars:
                if name.startswith(token) and not re.search(r"_(L|R)$", name):
                    self.lint_tree.insert("", "end", values=(name, "Latéralité manquante (`_L`/`_R`)", f"→ {name}_L ou {name}_R"))
                    break

            # Style PTV (recommandation)
            if s.structure_type == "PTV":
                if self.ptv_style.get() == "cGy" and s.rx:
                    cg = int(round(100 * s.rx))
                    sug = f"PTV_{cg:04d}"
                    if name != sug:
                        self.lint_tree.insert("", "end", values=(name, "Style PTV recommandé : cGy", f"→ {sug}"))
                elif self.ptv_style.get() == "Relatif":
                    rank = ptv_rank.get(s.name, None)
                    if rank is not None:
                        labels = ["PTV_High", "PTV_Mid", "PTV_Low"]
                        sug = labels[rank] if rank < 3 else f"PTV_Level{rank + 1}"
                        if name != sug:
                            self.lint_tree.insert("", "end", values=(name, "Style PTV recommandé : Relatif", f"→ {sug}"))

    # ------------------------------------------------------ DVH & indices ---
    def _build_dvh(self, parent: tk.Widget) -> None:
        """Construit l’onglet DVH (chargement CSV, indices et tracé)."""
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=6, padx=8)
        ttk.Button(top, text="Charger CSV DVH…", command=self._load_dvh).pack(side="left")
        ttk.Button(top, text="Calculer indices", command=self._calc_indices).pack(side="left", padx=8)

        padf = ttk.LabelFrame(parent, text="Indice de Paddick (volumes en cm³)")
        padf.pack(fill="x", padx=8, pady=8)
        self.tv_var = tk.DoubleVar(value=0.0)
        self.piv_var = tk.DoubleVar(value=0.0)
        self.int_var = tk.DoubleVar(value=0.0)
        ttk.Label(padf, text="TV [cm³] :").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Entry(padf, textvariable=self.tv_var, width=10).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(padf, text="PIV [cm³] :").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        ttk.Entry(padf, textvariable=self.piv_var, width=10).grid(row=0, column=3, padx=4, pady=4)
        ttk.Label(padf, text="TV∩PIV [cm³] :").grid(row=0, column=4, padx=4, pady=4, sticky="e")
        ttk.Entry(padf, textvariable=self.int_var, width=10).grid(row=0, column=5, padx=4, pady=4)
        ttk.Button(padf, text="Calculer CI", command=self._calc_paddick).grid(row=0, column=6, padx=6, pady=4)
        self.ci_var = tk.StringVar(value="—")
        ttk.Label(padf, text="CI [—] :").grid(row=0, column=7, sticky="e")
        ttk.Label(padf, textvariable=self.ci_var).grid(row=0, column=8, sticky="w")

        rf = ttk.Frame(parent)
        rf.pack(fill="both", expand=True, padx=8, pady=8)
        cols = (
            "Structure",
            "Type",
            "Rx [Gy]",
            "D2% [Gy]",
            "D5% [Gy]",
            "D10% [Gy]",
            "D50% [Gy]",
            "D90% [Gy]",
            "D95% [Gy]",
            "D98% [Gy]",
            "V95%Rx [%]",
            "V100%Rx [%]",
            "HI [—]",
            "gEUD(a) [Gy]",
        )
        self.res_tree = ttk.Treeview(rf, columns=cols, show="headings", height=16)
        for c in cols:
            self.res_tree.heading(c, text=c)
            self.res_tree.column(c, width=110 if c not in ("Structure", "Type") else (170 if c == "Structure" else 80), anchor="center")
        self.res_tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(rf, orient="vertical", command=self.res_tree.yview)
        vsb.pack(side="right", fill="y")
        self.res_tree.configure(yscroll=vsb.set)

        pf = ttk.LabelFrame(parent, text="Tracer DVH")
        pf.pack(fill="x", padx=8, pady=6)
        self.plot_name = tk.StringVar()
        ttk.Label(pf, text="Structure :").pack(side="left", padx=4)
        ttk.Entry(pf, textvariable=self.plot_name, width=18).pack(side="left")
        ttk.Button(pf, text="Tracer", command=self._plot_dvh).pack(side="left", padx=6)

    def _load_dvh(self) -> None:
        """Charge un CSV DVH cumulatif et prépare les colonnes standard."""
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.dvh_df = _prep_dvh(df)
            messagebox.showinfo(
                "OK",
                f"DVH chargé : {len(self.dvh_df)} points, {self.dvh_df['Structure'].nunique()} structures.",
            )
        except Exception as exc:
            messagebox.showerror("Erreur", f"Lecture DVH échouée :\n{exc}")

    def _calc_indices(self) -> None:
        """Calcule des indices DVH (ICRU, gEUD, V95/100%Rx, etc.) et alimente la table."""
        if self.dvh_df is None or self.dvh_df.empty:
            messagebox.showerror("Erreur", "Chargez d'abord un CSV DVH.")
            return
        self.res_tree.delete(*self.res_tree.get_children())

        stype_map: Dict[str, str] = {s.name: s.structure_type for s in self.structures}
        rx_map: Dict[str, Optional[float]] = {s.name: s.rx for s in self.structures}
        a_map: Dict[str, Optional[float]] = {s.name: s.a_param for s in self.structures}

        for sname, grp in self.dvh_df.groupby("Structure"):
            dvh = grp[["DoseGy", "Vrel"]].reset_index(drop=True)
            d2 = d_at_v(dvh, 0.02)
            d5 = d_at_v(dvh, 0.05)
            d10 = d_at_v(dvh, 0.10)
            d50 = d_at_v(dvh, 0.50)
            d90 = d_at_v(dvh, 0.90)
            d95 = d_at_v(dvh, 0.95)
            d98 = d_at_v(dvh, 0.98)

            stype = stype_map.get(sname, "—")
            rx = rx_map.get(sname, None)
            v95rx = v100rx = None
            if stype == "PTV" and rx and rx > 0:
                v95rx = v_at_d(dvh, 0.95 * rx)
                v100rx = v_at_d(dvh, rx)
            hi = hi_icru(d2, d98, rx) if stype == "PTV" else None

            a = a_map.get(sname, None)
            geud_val = geud_from_cum_dvh(dvh, a) if a is not None else None

            self.res_tree.insert(
                "",
                "end",
                values=(
                    sname,
                    stype,
                    format_value(rx),
                    format_value(d2),
                    format_value(d5),
                    format_value(d10),
                    format_value(d50),
                    format_value(d90),
                    format_value(d95),
                    format_value(d98),
                    format_value(100 * v95rx) if v95rx is not None else "—",
                    format_value(100 * v100rx) if v100rx is not None else "—",
                    format_value(hi),
                    format_value(geud_val),
                ),
            )

    def _plot_dvh(self) -> None:
        """Trace la courbe DVH cumulatif pour la structure demandée."""
        name = self.plot_name.get().strip()
        if not name:
            messagebox.showerror("Erreur", "Indiquez le nom de la structure.")
            return
        if self.dvh_df is None or self.dvh_df.empty:
            messagebox.showerror("Erreur", "Chargez d'abord un CSV DVH.")
            return
        grp = self.dvh_df[self.dvh_df["Structure"] == name]
        if grp.empty:
            messagebox.showerror("Erreur", f"Structure '{name}' introuvable dans le DVH.")
            return
        dvh = grp[["DoseGy", "Vrel"]].reset_index(drop=True)
        plt.figure()
        plt.plot(dvh["DoseGy"].values, 100 * dvh["Vrel"].values, linewidth=2)
        plt.xlabel("Dose [Gy]")
        plt.ylabel("V(≥D) [%]")
        plt.grid(True)
        plt.title(f"DVH cumulatif – {name}")
        plt.show()

    def _calc_paddick(self) -> None:
        """Calcule l’indice de Paddick à partir des volumes saisis."""
        try:
            tv = float(self.tv_var.get())
            piv = float(self.piv_var.get())
            inter = float(self.int_var.get())
        except Exception:
            messagebox.showerror("Erreur", "Entrées invalides (TV/PIV/TV∩PIV).")
            return
        if tv <= 0 or piv <= 0:
            messagebox.showerror("Erreur", "TV et PIV doivent être > 0.")
            return
        ci = (inter * inter) / (tv * piv)
        self.ci_var.set(format_value(ci, 3))


# ============================================================================
# Entrée programme
# ============================================================================

def main() -> None:
    """
    Point d’entrée de l’application.

    Initialise et lance la boucle Tkinter.
    """
    app = DosimApp()
    app.mainloop()


if __name__ == "__main__":
    main()
