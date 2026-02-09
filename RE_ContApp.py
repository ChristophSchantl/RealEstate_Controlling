# app.py
# Streamlit: PDF-Gantt (MS Project/ProjectLibre Export) -> Schedule Checks + (vereinfachtes) CPM + Finance/Carry
# Run:
#   pip install streamlit pandas numpy plotly camelot-py[cv]
#   streamlit run app.py

import io
import re
import os
import math
import tempfile
import datetime as dt
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ----------------------------
# Helpers: date / duration
# ----------------------------
def parse_date_any(x: Any) -> pd.Timestamp:
    """
    Parses dates like:
      'lun 10/03/25', 'mar 25/11/25', '10/03/2025', '2025-03-10'
    Returns pd.Timestamp (date normalized) or NaT.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return pd.NaT
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return pd.NaT

    # 1) Try dd/mm/yy or dd/mm/yyyy anywhere in string
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return pd.Timestamp(dt.date(y, mo, d))
        except Exception:
            return pd.NaT

    # 2) ISO fallback
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return pd.Timestamp(ts.date())
    except Exception:
        return pd.NaT


def parse_duration_days(x: Any) -> float:
    """
    Extracts a numeric duration in days from strings like:
      '435,88 días', '6 días', '0 días'
    Returns float or NaN.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if not s or s == "nan":
        return np.nan
    s = s.replace(",", ".")
    m = re.search(r"([-+]?\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


# ----------------------------
# Predecessors parsing
# ----------------------------
def parse_predecessors_cell(s: Any) -> List[Tuple[int, str, int]]:
    """
    Parses predecessor strings like:
      "57FC+1 día"  -> (57, 'FS', +1)
      "155CC+15 días" -> (155, 'SS', +15)
      "213FF-20 días" -> (213, 'FF', -20)
      "75" -> (75, 'FS', 0)
      "32FC+1 día; 45CC+2 días" -> list

    Relations mapping (Spanish abbreviations typically from MS Project):
      FC -> FS (Finish-to-Start)
      CC -> SS (Start-to-Start)
      FF -> FF
      SF -> SF
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    txt = str(s).strip()
    if not txt or txt.lower() == "nan":
        return []

    parts = [p.strip() for p in re.split(r"[;,]\s*", txt) if p.strip()]
    out: List[Tuple[int, str, int]] = []

    for p in parts:
        mid = re.search(r"(\d+)", p)
        if not mid:
            continue
        pred_id = int(mid.group(1))

        rel = "FS"
        if "CC" in p:
            rel = "SS"
        elif "FC" in p:
            rel = "FS"
        elif "FF" in p:
            rel = "FF"
        elif "SF" in p:
            rel = "SF"

        lag = 0
        mlag = re.search(r"([+-]\s*\d+)", p)
        if mlag:
            lag = int(mlag.group(1).replace(" ", ""))

        out.append((pred_id, rel, lag))

    return out


# ----------------------------
# CPM (simplified, day-based)
# ----------------------------
def build_graph(df: pd.DataFrame) -> Tuple[List[int], List[Tuple[int, int, str, int]]]:
    """
    Nodes: task Ids
    Edges: (pred, succ, rel, lag)
    """
    nodes = df["Id"].astype(int).tolist()
    node_set = set(nodes)
    edges: List[Tuple[int, int, str, int]] = []
    for _, r in df.iterrows():
        tid = int(r["Id"])
        for pred_id, rel, lag in r["pred_parsed"]:
            if pred_id in node_set:
                edges.append((pred_id, tid, rel, lag))
    return nodes, edges


def topo_sort(nodes: List[int], edges: List[Tuple[int, int, str, int]]) -> List[int]:
    succ: Dict[int, List[int]] = {n: [] for n in nodes}
    indeg: Dict[int, int] = {n: 0 for n in nodes}
    for a, b, _, _ in edges:
        succ[a].append(b)
        indeg[b] += 1
    q = sorted([n for n in nodes if indeg[n] == 0])
    order: List[int] = []
    while q:
        n = q.pop(0)
        order.append(n)
        for b in succ.get(n, []):
            indeg[b] -= 1
            if indeg[b] == 0:
                q.append(b)
                q.sort()
    # If cycle exists, fall back to original order
    if len(order) != len(nodes):
        return nodes
    return order


def compute_cpm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified CPM in numeric "days". Uses duration and predecessor constraints:
      FS, SS, FF, SF with lag (days).
    Returns DataFrame: Id, ES, EF, LS, LF, total_float
    """
    nodes, edges = build_graph(df)
    dur = df.set_index("Id")["dur_days"].fillna(0).to_dict()

    # adjacency
    succ: Dict[int, List[Tuple[int, str, int]]] = {n: [] for n in nodes}
    pred: Dict[int, List[Tuple[int, str, int]]] = {n: [] for n in nodes}
    indeg: Dict[int, int] = {n: 0 for n in nodes}

    for a, b, rel, lag in edges:
        succ[a].append((b, rel, lag))
        pred[b].append((a, rel, lag))
        indeg[b] += 1

    order = topo_sort(nodes, edges)

    ES: Dict[int, float] = {n: 0.0 for n in nodes}
    EF: Dict[int, float] = {n: 0.0 for n in nodes}

    # Forward pass
    for n in order:
        EF[n] = ES[n] + float(dur.get(n, 0.0))
        for b, rel, lag in succ.get(n, []):
            db = float(dur.get(b, 0.0))
            if rel == "FS":
                ES[b] = max(ES[b], EF[n] + lag)
            elif rel == "SS":
                ES[b] = max(ES[b], ES[n] + lag)
            elif rel == "FF":
                ES[b] = max(ES[b], EF[n] + lag - db)
            elif rel == "SF":
                ES[b] = max(ES[b], ES[n] + lag - db)

    proj = max(EF.values()) if EF else 0.0

    LS: Dict[int, float] = {n: proj - float(dur.get(n, 0.0)) for n in nodes}
    LF: Dict[int, float] = {n: proj for n in nodes}

    # Backward pass
    for n in reversed(order):
        LF[n] = LS[n] + float(dur.get(n, 0.0))
        for a, rel, lag in pred.get(n, []):
            da = float(dur.get(a, 0.0))
            if rel == "FS":
                # ES[n] >= EF[a] + lag  -> LS[a] <= LS[n] - lag - da
                LS[a] = min(LS[a], LS[n] - lag - da)
            elif rel == "SS":
                # ES[n] >= ES[a] + lag -> LS[a] <= LS[n] - lag
                LS[a] = min(LS[a], LS[n] - lag)
            elif rel == "FF":
                # EF[n] >= EF[a] + lag -> LS[a] <= LF[n] - lag - da
                LS[a] = min(LS[a], LF[n] - lag - da)
            elif rel == "SF":
                # EF[n] >= ES[a] + lag -> LS[a] <= LF[n] - lag
                LS[a] = min(LS[a], LF[n] - lag)

    TF = {n: max(0.0, LS[n] - ES[n]) for n in nodes}

    out = pd.DataFrame(
        {
            "Id": nodes,
            "ES": [ES[n] for n in nodes],
            "EF": [EF[n] for n in nodes],
            "LS": [LS[n] for n in nodes],
            "LF": [LF[n] for n in nodes],
            "total_float": [TF[n] for n in nodes],
        }
    )
    return out


# ----------------------------
# Finance: simple spend curve + carry
# ----------------------------
def build_cashflow(df: pd.DataFrame, total_budget: float, annual_rate: float, debt_share: float) -> pd.DataFrame:
    """
    Creates daily spend curve by distributing each task budget evenly between start-finish days.
    Task budget allocation: proportional to duration (placeholder, replace with real cost codes later).
    Carry: interest on cumulative deployed debt.
    """
    df2 = df.dropna(subset=["start", "finish"]).copy()
    if df2.empty or total_budget <= 0:
        return pd.DataFrame(columns=["date", "daily_spend", "cum_spend", "daily_carry", "cum_carry"])

    start = df2["start"].min()
    end = df2["finish"].max()
    days = pd.date_range(start, end, freq="D")

    weights = df2["dur_days"].fillna(0).clip(lower=0).astype(float).to_numpy()
    wsum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
    df2["task_budget"] = total_budget * (weights / wsum)

    spend = pd.Series(0.0, index=days)
    for _, r in df2.iterrows():
        d0 = pd.to_datetime(r["start"])
        d1 = pd.to_datetime(r["finish"])
        rng = pd.date_range(d0, d1, freq="D")
        if len(rng) == 0:
            continue
        spend.loc[rng] += float(r["task_budget"]) / float(len(rng))

    cum_spend = spend.cumsum()
    debt = cum_spend * float(debt_share)
    daily_rate = float(annual_rate) / 365.0
    daily_carry = debt * daily_rate
    cum_carry = daily_carry.cumsum()

    return pd.DataFrame(
        {
            "date": days,
            "daily_spend": spend.values,
            "cum_spend": cum_spend.values,
            "daily_carry": daily_carry.values,
            "cum_carry": cum_carry.values,
        }
    )


def delay_carry_impact(cf: pd.DataFrame, annual_rate: float, delay_days: int, basis: str = "debt") -> float:
    """
    Extra carry if project is delayed by N days and capital stays deployed.
    basis:
      - "debt": carry on deployed debt
      - "capital": carry on deployed total capital (debt+equity) -> proxy/opportunity cost
    """
    if cf.empty or delay_days <= 0:
        return 0.0

    last_cum = float(cf["cum_spend"].iloc[-1])
    daily_rate = float(annual_rate) / 365.0
    if basis == "capital":
        return last_cum * daily_rate * float(delay_days)
    # default: debt carry already uses debt_share in CF; here use implied last debt from CF carry slope:
    # we can approximate last debt as last daily_carry / daily_rate
    last_daily_carry = float(cf["daily_carry"].iloc[-1])
    last_debt = last_daily_carry / daily_rate if daily_rate > 0 else 0.0
    return last_debt * daily_rate * float(delay_days)


# ----------------------------
# PDF -> tables (Camelot)
# ----------------------------
def extract_tables(pdf_bytes: bytes, pages: str, flavor: str) -> List[pd.DataFrame]:
    """
    Camelot extraction. Works best on text PDFs with table lines (lattice) or whitespace (stream).
    """
    try:
        import camelot  # type: ignore
    except Exception as e:
        raise RuntimeError("camelot ist nicht installiert. Installiere: pip install camelot-py[cv]") from e

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name

    try:
        tables = camelot.read_pdf(path, pages=pages, flavor=flavor)
        out: List[pd.DataFrame] = []
        for t in tables:
            df = t.df.copy()
            df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
            # Try header detection
            if len(df) > 1:
                first = df.iloc[0].astype(str)
                if first.str.contains("Duraci|Comienzo|Fin|Predeces|Nombre|tarea|Id", case=False, regex=True).any():
                    df.columns = df.iloc[0]
                    df = df.iloc[1:].reset_index(drop=True)
            out.append(df)
        return out
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ----------------------------
# Normalization pipeline
# ----------------------------
def normalize_schedule(df_raw: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    mapping maps raw column names -> canonical:
      Id, EDT(optional), Nombre de tarea, Duración, Comienzo, Fin, Predecesoras
    """
    df = df_raw.copy()

    # apply rename
    rename = {raw: canon for raw, canon in mapping.items() if raw in df.columns and canon}
    df = df.rename(columns=rename)

    # ensure columns exist
    for col in ["Id", "Nombre de tarea", "Duración", "Comienzo", "Fin", "Predecesoras"]:
        if col not in df.columns:
            df[col] = np.nan
    if "EDT" not in df.columns:
        df["EDT"] = ""

    # clean Id
    df["Id"] = df["Id"].astype(str).str.extract(r"(\d+)", expand=False)
    df = df.dropna(subset=["Id"]).copy()
    df["Id"] = df["Id"].astype(int)

    # dates + duration
    df["start"] = df["Comienzo"].apply(parse_date_any)
    df["finish"] = df["Fin"].apply(parse_date_any)
    df["dur_days"] = df["Duración"].apply(parse_duration_days)

    # predecessor parsing
    df["Predecesoras"] = df["Predecesoras"].astype(str).replace("nan", "", regex=False)
    df["pred_parsed"] = df["Predecesoras"].apply(parse_predecessors_cell)

    # task name cleanup
    df["Nombre de tarea"] = df["Nombre de tarea"].astype(str).str.strip()

    return df


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF Gantt Controlling", layout="wide")
st.title("PDF → Gantt Controlling (Schedule + Finance)")

with st.sidebar:
    st.header("Input")
    pdf = st.file_uploader("Gantt PDF hochladen", type=["pdf"])
    pages = st.text_input("Seiten (Camelot) z.B. 1-6 oder 1,2,5", value="1-10")
    flavor = st.selectbox("PDF Tabellenmodus", ["lattice (mit Linien)", "stream (ohne Linien)"], index=0)
    flavor_val = "lattice" if flavor.startswith("lattice") else "stream"

    st.header("Flags")
    loe_threshold = st.number_input("LOE/Teppich-Flag ab Dauer (Tage)", min_value=30, value=250, step=10)
    flag_zero_day = st.checkbox("0-Tage Tasks flaggen", value=True)
    dup_name_flag = st.checkbox("Doppelte Task-Namen flaggen", value=True)

    st.header("Finance")
    total_budget = st.number_input("Total Budget (EUR)", min_value=0.0, value=0.0, step=10000.0, format="%.2f")
    annual_rate_pct = st.number_input("Finanzierungszins p.a. (%)", min_value=0.0, value=10.0, step=0.25, format="%.2f")
    debt_share_pct = st.slider("Debt Anteil (%)", min_value=0, max_value=100, value=50, step=5)
    delay_days = st.number_input("Was-wäre-wenn Verzug (Tage)", min_value=0, value=30, step=5)
    delay_basis = st.selectbox("Delay Carry-Basis", ["debt (Zins)", "capital (Opportunity)"], index=0)

if not pdf:
    st.info("PDF hochladen → Tabellen extrahieren → Spalten mappen → Analyse.")
    st.stop()

pdf_bytes = pdf.read()

try:
    tables = extract_tables(pdf_bytes, pages=pages, flavor=flavor_val)
except Exception as e:
    st.error(f"PDF-Extraktion fehlgeschlagen: {e}")
    st.stop()

if not tables:
    st.error("Keine Tabellen erkannt. Wenn das PDF ein Scan/Bild ist, brauchst du OCR.")
    st.stop()

st.subheader("1) Tabellen gefunden")
st.write(f"Anzahl erkannter Tabellen: **{len(tables)}**")
table_idx = st.selectbox("Welche Tabelle verwenden?", list(range(len(tables))), index=0)
df_raw = tables[table_idx].copy()

st.dataframe(df_raw, use_container_width=True, height=320)

st.subheader("2) Spalten-Mapping (einmal sauber setzen)")
cols = list(df_raw.columns)

def _pick(label: str, default_idx: int) -> str:
    if not cols:
        return ""
    idx = min(default_idx, len(cols) - 1)
    return st.selectbox(label, cols, index=idx)

col_id = _pick("ID-Spalte", 0)
col_task = _pick("Task-Name", 3 if len(cols) > 3 else 1)
col_dur = _pick("Dauer", 4 if len(cols) > 4 else 2)
col_start = _pick("Start (Comienzo)", 5 if len(cols) > 5 else 3)
col_finish = _pick("Finish (Fin)", 6 if len(cols) > 6 else 4)
col_pred = _pick("Predecessors (Predecesoras)", 7 if len(cols) > 7 else 5)
col_wbs = st.selectbox("WBS/EDT optional", ["(none)"] + cols, index=0)

mapping = {
    col_id: "Id",
    col_task: "Nombre de tarea",
    col_dur: "Duración",
    col_start: "Comienzo",
    col_finish: "Fin",
    col_pred: "Predecesoras",
}
if col_wbs != "(none)":
    mapping[col_wbs] = "EDT"

run = st.button("✅ Normalisieren & Analysieren")

if not run:
    st.stop()

df = normalize_schedule(df_raw, mapping=mapping)

# CPM
cpm = compute_cpm(df)
df = df.merge(cpm, on="Id", how="left")
df["is_critical"] = df["total_float"].fillna(999999.0) <= 0.5

# Flags
df["flag_loe"] = df["dur_days"].fillna(0.0) >= float(loe_threshold)
df["flag_zero_day"] = df["dur_days"].fillna(0.0) == 0.0

df["task_name_norm"] = df["Nombre de tarea"].astype(str).str.lower().str.strip()
df["flag_dup_name"] = df["task_name_norm"].duplicated(keep=False) if dup_name_flag else False

# Finance
annual_rate = float(annual_rate_pct) / 100.0
debt_share = float(debt_share_pct) / 100.0
cf = build_cashflow(df, total_budget=float(total_budget), annual_rate=annual_rate, debt_share=debt_share)
extra_delay = delay_carry_impact(cf, annual_rate=annual_rate, delay_days=int(delay_days), basis="capital" if delay_basis.startswith("capital") else "debt")

# ----------------------------
# Output tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Schedule Health", "Timeline / Critical Path", "Finance / Carry"])

with tab1:
    st.subheader("Explizite Problempunkte (aus PDF extrahiert)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tasks", len(df))
    c2.metric("Critical (Float≈0)", int(df["is_critical"].sum()))
    c3.metric("0-Tage", int(df["flag_zero_day"].sum()) if flag_zero_day else 0)
    c4.metric(f"LOE ≥ {loe_threshold}d", int(df["flag_loe"].sum()))
    c5.metric("Doppelte Namen", int(df["flag_dup_name"].sum()) if dup_name_flag else 0)

    show_cols = [
        "Id", "EDT", "Nombre de tarea", "Duración", "Comienzo", "Fin", "Predecesoras",
        "dur_days", "flag_zero_day", "flag_loe", "flag_dup_name", "is_critical", "total_float"
    ]
    existing = [c for c in show_cols if c in df.columns]
    # Sort to bubble up red flags
    sort_cols = ["is_critical", "flag_zero_day", "flag_loe", "flag_dup_name"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    st.dataframe(
        df[existing].sort_values(sort_cols, ascending=False),
        use_container_width=True,
        height=520
    )

    st.download_button(
        "⬇️ Export: normalisierte CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="schedule_normalized.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Gantt (aus Start/Finish) + Critical Markierung")
    plot_df = df.dropna(subset=["start", "finish"]).copy()
    if plot_df.empty:
        st.warning("Keine gültigen Start/Finish-Daten erkannt (Mapping/Extraktion prüfen).")
    else:
        plot_df["critical_label"] = np.where(plot_df["is_critical"], "CRITICAL", "non-critical")
        # For performance, limit very large tables
        max_rows = 400
        if len(plot_df) > max_rows:
            st.warning(f"Viele Tasks ({len(plot_df)}). Anzeige auf {max_rows} gekürzt (Performance).")
            plot_df = plot_df.head(max_rows)

        fig = px.timeline(
            plot_df,
            x_start="start",
            x_end="finish",
            y="Nombre de tarea",
            color="critical_label",
            hover_data=["Id", "EDT", "Duración", "Predecesoras", "total_float"],
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Finance / Carry (Investor View)")
    st.caption("Budget-Verteilung hier: **proportional zur Dauer** (Placeholder). Für echtes Controlling ersetzt du das durch Cost Codes / BOQ je Task.")

    if cf.empty:
        st.info("Keine Finance-Kurve: entweder Budget=0 oder keine gültigen Start/Finish-Daten.")
    else:
        base_carry = float(cf["cum_carry"].iloc[-1])
        st.metric("Carry Basisplan (EUR)", f"{base_carry:,.0f}")
        st.metric(f"Extra Carry bei +{int(delay_days)} Tagen (EUR)", f"{extra_delay:,.0f}")
        st.metric("Carry Basisplan + Delay (EUR)", f"{(base_carry + extra_delay):,.0f}")

        fig2 = px.line(cf, x="date", y=["cum_spend", "cum_carry"], markers=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "⬇️ Export: Cashflow CSV",
            cf.to_csv(index=False).encode("utf-8"),
            file_name="cashflow.csv",
            mime="text/csv",
        )

st.success("Fertig. Wenn die Tabellen leer/kaputt aussehen: Flavor wechseln (lattice/stream) oder PDF ist ein Scan → OCR nötig.")
