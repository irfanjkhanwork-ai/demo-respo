import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Utilities ----------
def daterange_days(a, b):
    return (pd.to_datetime(b) - pd.to_datetime(a)).dt.days

def load_docs(doc_dir="docs"):
    paths = sorted(Path(doc_dir).glob("*.txt"))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return paths, texts

def top_k_docs(query, paths, texts, k=2):
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts + [query])
    sim = cosine_similarity(X[-1], X[:-1]).flatten()
    idx = sim.argsort()[::-1][:k]
    return [(paths[i].name, texts[i].strip(), float(sim[i])) for i in idx if sim[i] > 0.0]

def recommend_action(supplier_delays: pd.DataFrame):
    # naive heuristic for demo
    if supplier_delays.empty or len(supplier_delays) < 2:
        return None
    top = supplier_delays.iloc[0]
    second = supplier_delays.iloc[1]
    if top['avg_delay_days'] >= 2 and (top['avg_delay_days'] - second['avg_delay_days'])/max(second['avg_delay_days'], 0.1) >= 0.3:
        return {
            "action_id": "act_shift_10pct",
            "explain": f"Shift 10% orders from {top['supplier']} to {second['supplier']} to mitigate lateness.",
            "cost_benefit": {"cost": 4800, "benefit": 9200, "currency": "USD"},
            "params": {"from": top['supplier'], "to": second['supplier'], "percent": 10}
        }
    return None

def apply_whatif(df: pd.DataFrame, cmd: str) -> pd.DataFrame:
    """
    Supports: /whatif shift 10% S47->S12
    Reassigns ~10% of S47 orders to S12 deterministically by order_id modulo.
    """
    try:
        tokens = cmd.strip().split()
        if len(tokens) < 4 or tokens[0] != "/whatif" or tokens[1] != "shift":
            return df
        pct_token = tokens[2]
        route = tokens[3]
        if not pct_token.endswith("%") or "->" not in route:
            return df
        pct = float(pct_token[:-1]) / 100.0
        src, dst = route.split("->")
        src, dst = src.strip(), dst.strip()
        if src not in df["supplier"].unique():
            return df
        df2 = df.copy()
        df2["order_id"] = df2["order_id"].astype(int)
        step = max(1, int(round(1/pct)))
        mask = (df2["supplier"] == src) & ((df2["order_id"] % step) == 0)
        df2.loc[mask, "supplier"] = dst
        return df2
    except Exception:
        return df

# ---------- Page ----------
st.set_page_config(page_title="Decision Intelligence – 1-Hour Demo", layout="wide")

st.title("Reasoning-First Ops (1-Hour Demo)")
st.caption("Query → Reasoning → Insight → Traceability")

# Input
query = st.text_input("Ask a question", value="Why are shipments late this week?")

# Data
csv_path = Path("data/orders.csv")
if not csv_path.exists():
    st.error("Missing data/orders.csv")
    st.stop()

df = pd.read_csv(csv_path)
df['delay_days'] = daterange_days(df['promised_date'], df['actual_date'])

# Optional: /whatif minimal support
whatif_note = ""
if query.startswith("/whatif"):
    df = apply_whatif(df, query)
    whatif_note = "Projected: 10% load rebalanced per /whatif."

# Reasoning Step 1: SQL-style aggregation with DuckDB
con = duckdb.connect(database=':memory:')
con.register('orders', df)
sql = """
SELECT supplier,
       ROUND(AVG(delay_days), 2) AS avg_delay_days,
       COUNT(*) AS orders
FROM orders
GROUP BY supplier
ORDER BY avg_delay_days DESC
"""
supplier_delays = con.execute(sql).df()

# Reasoning Step 2: Retrieve doc evidence (TF-IDF)
paths, texts = load_docs("docs")
evidence = top_k_docs(query, paths, texts, k=2)

# Synthesis -> Insight Object (deterministic, demo-friendly)
if not supplier_delays.empty:
    top = supplier_delays.iloc[0]
    baseline = supplier_delays['avg_delay_days'].mean()
    lift_pct = 0.0 if baseline == 0 else (top['avg_delay_days'] - baseline)/max(baseline, 0.1)
    confidence = max(0.55, min(0.9, 0.65 + 0.2 * (lift_pct)))
    finding = f"Supplier {top['supplier']} is the current bottleneck with average delay {top['avg_delay_days']} days."
else:
    finding = "Insufficient data to determine a bottleneck."
    confidence = 0.6

impact_projection = {
    "type": "throughput",
    "unit": "orders/day",
    "delta": int(round(-12 * supplier_delays.iloc[0]['avg_delay_days'])) if not supplier_delays.empty else 0,
    "horizon_days": 14
}

insight = {
    "id": f"ins_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    "finding": finding,
    "rationale": [
        {"type": "data", "detail": "SQL aggregation of avg delay by supplier"},
        *([{"type": "doc", "detail": f"{name} (similarity {sim:.2f})"} for name, _, sim in evidence] if evidence else [])
    ],
    "confidence": round(confidence, 2),
    "impact": impact_projection,
    "data_lineage": ["source:data/orders.csv"],
    "model_lineage": {"planner":"demo-v0", "retrieval":"tfidf", "synth":"template-v0"}
}

# ---------- UI Layout ----------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Insight")
    st.markdown(f"**{insight['finding']}**")
    st.progress(min(1.0, insight["confidence"]))
    subtitle = f"Confidence: {insight['confidence']}  •  Impact: {insight['impact']['delta']} {insight['impact']['unit']} over {insight['impact']['horizon_days']} days"
    if whatif_note:
        subtitle += f"  •  {whatif_note}  •  Projected OTIF +3–5%"
    st.caption(subtitle)

    st.markdown("#### Delay by supplier")
    st.dataframe(supplier_delays, use_container_width=True, hide_index=True)
    st.bar_chart(supplier_delays.set_index("supplier")["avg_delay_days"])

    action = recommend_action(supplier_delays)
    if action:
        st.markdown("#### Recommended action")
        st.write(f"**{action['explain']}**")
        st.json(action)
        if st.button("Mark as executed"):
            st.success("Action logged (demo).")

with right:
    st.subheader("Traceability")
    st.markdown("**Rationale**")
    for r in insight["rationale"]:
        st.write(f"- {r['type']}: {r['detail']}")
    with st.expander("Show SQL"):
        st.code(sql, language="sql")
    with st.expander("Preview data sample"):
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    with st.expander("Doc evidence"):
        if evidence:
            for name, text, sim in evidence:
                st.markdown(f"**{name}** (similarity {sim:.2f})")
                st.code(text[:800] + ("..." if len(text) > 800 else ""))
        else:
            st.caption("No matching documents.")

st.divider()
st.caption(f"Insight ID: {insight['id']}  •  Data sources: {', '.join(insight['data_lineage'])}")
EOF
