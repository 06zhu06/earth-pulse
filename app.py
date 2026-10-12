from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EARTH PULSE", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = Path(__file__).with_name("database.csv")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["Date"].where(frame["Date"].str.contains("T"), frame["Date"] + " " + frame["Time"]), errors="coerce", format="mixed", utc=True)
    frame["DateOnly"] = frame["timestamp"].dt.date
    frame["year"] = frame["timestamp"].dt.year.astype(int)
    frame["month"] = frame["timestamp"].dt.to_period("M").astype(str)
    frame["Type"] = frame["Type"].fillna("Unknown")
    frame["Magnitude Type"] = frame["Magnitude Type"].fillna("Unknown")
    for col in ["Depth", "Magnitude", "Latitude", "Longitude"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp", "Latitude", "Longitude", "Depth", "Magnitude"]).sort_values("timestamp").reset_index(drop=True)

df = load_data(str(DATA_PATH))
min_date, max_date = df["DateOnly"].min(), df["DateOnly"].max()

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 75% 0%, #142c42 0%, #07121f 38%, #040910 100%); color: #e7f3ff;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background: #071522; border-right: 1px solid #17344a;}
h1 {letter-spacing: .18em; font-size: 2.4rem !important; color: #f2f8ff;}
h2, h3 {color: #d6efff;}
[data-testid="stMetricValue"] {color: #5ce1e6;}
[data-testid="stMetricLabel"] {color: #91a9bb;}
.pulse-kicker {color:#58e0dc;letter-spacing:.26em;font-size:.8rem;font-weight:700;}
.pulse-subtitle {color:#b9cbd8;font-size:1.05rem;margin-bottom:1.2rem;}
.data-note {border-left:3px solid #ff9d42;padding:.55rem .8rem;background:rgba(27,57,77,.55);color:#b9cbd8;font-size:.86rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="pulse-kicker">EARTH PULSE / GLOBAL SEISMIC INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("全球地震活动监测与时空演化分析平台")
st.markdown('<div class="pulse-subtitle">Global Seismic Intelligence & Visualization System · 面向地震活动研究员</div>', unsafe_allow_html=True)
st.markdown('<div class="data-note">历史数据快照：1965–2016 · 来源：Kaggle / USGS Earthquake Database · 仅用于历史探索与研究分析，不提供实时预警或地震预测。</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### CONTROL ROOM")
    date_range = st.date_input("研究时间窗口", (min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    min_mag = st.slider("最低震级", float(df.Magnitude.min()), float(df.Magnitude.max()), 5.5, 0.1)
    depth_limit = st.slider("最大深度（km）", 0.0, float(df.Depth.quantile(.995)), float(df.Depth.quantile(.995)), 5.0)
    types = sorted(df.Type.unique().tolist())
    selected_types = st.multiselect("记录类型", types, default=["Earthquake"] if "Earthquake" in types else types)
    st.markdown("---")
    playback_unit = st.radio("回放粒度", ["年度", "月度"], horizontal=True)
    st.caption("地图点大小 = 震级；点颜色 = 震源深度（km）。颜色不是风险等级。")

filtered = df[(df.DateOnly >= start_date) & (df.DateOnly <= end_date) & (df.Magnitude >= min_mag) & (df.Depth <= depth_limit) & df.Type.isin(selected_types)].copy()
if playback_unit == "年度":
    periods = sorted(filtered.year.unique().tolist()) if not filtered.empty else sorted(df.year.unique().tolist())
    period_col, period_label = "year", "年度回放"
else:
    periods = sorted(filtered.month.unique().tolist()) if not filtered.empty else sorted(df.month.unique().tolist())
    period_col, period_label = "month", "月度回放"
if periods:
    step = st.slider(period_label, 0, len(periods)-1, len(periods)-1, key="timeline")
    current_period = periods[step]
    playback = filtered[filtered[period_col] <= current_period]
else:
    current_period, playback = "—", filtered

if filtered.empty:
    st.warning("当前筛选没有事件。请扩大时间范围、降低最低震级或增加记录类型。")

event_count = len(filtered)
max_mag = filtered.Magnitude.max() if event_count else 0
mean_mag = filtered.Magnitude.mean() if event_count else 0
shallow_count = int((filtered.Depth <= 70).sum()) if event_count else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("当前事件数", f"{event_count:,}")
c2.metric("最大震级", f"{max_mag:.1f}" if event_count else "—")
c3.metric("平均震级", f"{mean_mag:.2f}" if event_count else "—")
c4.metric("浅源事件 ≤70 km", f"{shallow_count:,}")

st.markdown(f"### 时空事件场 · {current_period} / 累计回放")
if not playback.empty:
    map_data = playback.copy()
    map_data["size"] = np.clip((map_data.Magnitude - 4.8) ** 2 * 2.2 + 5, 4, 90)
    map_data["hover"] = map_data.apply(lambda r: f"{r['DateOnly']} · M{r['Magnitude']:.1f} · 深度 {r['Depth']:.1f} km · {r['Type']}", axis=1)
    fig = go.Figure(go.Scattergeo(lon=map_data["Longitude"], lat=map_data["Latitude"], mode="markers", text=map_data["hover"], hovertemplate="%{text}<extra></extra>", marker=dict(size=map_data["size"], color=map_data["Depth"], colorscale="Turbo", cmin=0, cmax=max(700, depth_limit), colorbar=dict(title=dict(text="Depth (km)", font=dict(color="#c9d8e3")), tickfont=dict(color="#c9d8e3")), line=dict(width=.25, color="#ffffff"), opacity=.82)))
    fig.update_geos(showland=True, landcolor="#142331", showocean=True, oceancolor="#050d16", showcountries=True, countrycolor="#34566b", coastlinecolor="#4d758a", projection_type="natural earth", bgcolor="#050d16")
    fig.update_layout(height=570, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#050d16", plot_bgcolor="#050d16", font=dict(color="#d8e9f4"), showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

st.markdown("### 三维地震场")
st.caption("左图展示地球表面的全球事件位置；右图将震源深度作为第三维，便于识别空间聚集和深浅层结构。")
if not playback.empty:
    three_d = playback.copy()
    if len(three_d) > 3500:
        three_d = three_d.sample(3500, random_state=42)
    three_d["hover"] = three_d.apply(lambda r: f"{r['DateOnly']} · M{r['Magnitude']:.1f} · 深度 {r['Depth']:.1f} km · {r['Type']}", axis=1)
    three_d["point_size"] = np.clip((three_d["Magnitude"] - 4.8) ** 2 * 1.8 + 3, 3, 22)
    lat_rad = np.deg2rad(three_d["Latitude"])
    lon_rad = np.deg2rad(three_d["Longitude"])
    radius = 1.012
    three_d["x"] = radius * np.cos(lat_rad) * np.cos(lon_rad)
    three_d["y"] = radius * np.cos(lat_rad) * np.sin(lon_rad)
    three_d["z"] = radius * np.sin(lat_rad)
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(-np.pi / 2, np.pi / 2, 24)
    sphere_x = np.outer(np.cos(v), np.cos(u))
    sphere_y = np.outer(np.cos(v), np.sin(u))
    sphere_z = np.outer(np.sin(v), np.ones_like(u))
    globe_fig = go.Figure()
    globe_fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z, surfacecolor=np.zeros_like(sphere_x), colorscale=[[0, "#12324a"], [1, "#12324a"]], showscale=False, opacity=.88, hoverinfo="skip"))
    globe_fig.add_trace(go.Scatter3d(x=three_d["x"], y=three_d["y"], z=three_d["z"], mode="markers", text=three_d["hover"], hovertemplate="%{text}<extra></extra>", marker=dict(size=three_d["point_size"], color=three_d["Magnitude"], colorscale="Turbo", cmin=5.5, cmax=max(9.5, float(three_d["Magnitude"].max())), opacity=.9, line=dict(width=.2, color="#ffffff"))))
    globe_fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#050d16", scene=dict(aspectmode="cube", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="#050d16", camera=dict(eye=dict(x=1.55, y=1.55, z=1.1))), showlegend=False)

    space_fig = px.scatter_3d(three_d, x="Longitude", y="Latitude", z="Depth", color="Magnitude", size="point_size", hover_name="ID", hover_data={"Type": True, "Depth": ":.1f", "Magnitude": ":.1f", "point_size": False}, color_continuous_scale="Turbo", labels={"Longitude": "经度", "Latitude": "纬度", "Depth": "深度（km）", "Magnitude": "震级"})
    space_fig.update_traces(marker=dict(opacity=.86, line=dict(width=.2, color="#ffffff")))
    space_fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#050d16", scene=dict(bgcolor="#050d16", xaxis=dict(gridcolor="#25465a", zerolinecolor="#4a7087"), yaxis=dict(gridcolor="#25465a", zerolinecolor="#4a7087"), zaxis=dict(autorange="reversed", gridcolor="#25465a", zerolinecolor="#4a7087")), font_color="#cfe2ef", coloraxis_colorbar_title="震级", showlegend=False)
    globe_col, space_col = st.columns(2)
    with globe_col:
        st.plotly_chart(globe_fig, width="stretch", config={"displaylogo": False})
    with space_col:
        st.plotly_chart(space_fig, width="stretch", config={"displaylogo": False})

left, right = st.columns(2)
with left:
    st.markdown("### 活动趋势")
    trend = filtered.assign(period=filtered["timestamp"].dt.to_period("Y").astype(str)).groupby("period", as_index=False).agg(events=("ID", "count"), avg_magnitude=("Magnitude", "mean"))
    if not trend.empty:
        trend_fig = px.bar(trend, x="period", y="events", color="avg_magnitude", color_continuous_scale="Turbo", labels={"period":"年份", "events":"事件数", "avg_magnitude":"平均震级"})
        trend_fig.update_layout(height=330, paper_bgcolor="#071522", plot_bgcolor="#071522", font_color="#cfe2ef", coloraxis_colorbar_title="Avg M")
        st.plotly_chart(trend_fig, width="stretch", config={"displaylogo": False})
with right:
    st.markdown("### 震级 × 深度关系")
    if not filtered.empty:
        scatter_fig = px.scatter(filtered, x="Depth", y="Magnitude", color="Magnitude", size="Magnitude", hover_name="ID", color_continuous_scale="Turbo", labels={"Depth":"深度（km）", "Magnitude":"震级"})
        scatter_fig.update_layout(height=330, paper_bgcolor="#071522", plot_bgcolor="#071522", font_color="#cfe2ef", coloraxis_showscale=False)
        st.plotly_chart(scatter_fig, width="stretch", config={"displaylogo": False})

st.markdown("### 事件研究台")
if not filtered.empty:
    summary = filtered[["ID", "DateOnly", "Time", "Latitude", "Longitude", "Magnitude", "Depth", "Type", "Magnitude Type", "Source", "Status"]].sort_values("Magnitude", ascending=False)
    selected_id = st.selectbox("选择事件查看详情", summary.ID.head(100).tolist())
    selected = summary[summary.ID == selected_id].iloc[0]
    details = st.columns(5)
    details[0].metric("事件", selected.ID)
    details[1].metric("日期", str(selected.DateOnly))
    details[2].metric("位置", f"{selected.Latitude:.2f}, {selected.Longitude:.2f}")
    details[3].metric("震级 / 深度", f"M{selected.Magnitude:.1f} / {selected.Depth:.1f} km")
    details[4].metric("来源", selected.Source)
    st.dataframe(summary.head(200), width="stretch", hide_index=True)
    st.download_button("下载当前筛选事件 CSV", summary.to_csv(index=False).encode("utf-8"), "earth-pulse-events.csv", "text/csv")
else:
    st.info("暂无事件详情可显示。")

st.caption("说明：事件密度不等同于震害风险；缺少人口、建筑和损失数据，因此本平台只做地震活动描述性分析。")



