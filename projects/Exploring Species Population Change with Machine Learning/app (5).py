import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import pydeck as pdk

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Biodiversity Intelligence System", layout="wide")

st.title("🌍 Biodiversity Intelligence Dashboard (LIVE GBIF)")
st.markdown("Real-time global biodiversity monitoring & analysis system")

# =========================
# SPECIES LIST
# =========================
species_list = [
    "Panthera leo",
    "Panthera tigris",
    "Elephas maximus",
    "Canis lupus",
    "Ursus arctos",
    "Giraffa camelopardalis",
    "Bos taurus",
    "Homo sapiens",
    "Equus zebra",
    "Felis catus"
]

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("🔎 Control Panel")

selected_species = st.sidebar.selectbox("Choose Species", species_list)

custom_species = st.sidebar.text_input("Or Search Manually")

if custom_species:
    selected_species = custom_species

compare_species = st.sidebar.multiselect("Compare Species", species_list)

# =========================
# LIVE DATA (GBIF)
# =========================
@st.cache_data(ttl=3600)
def load_data(species_name):
    url = f"https://api.gbif.org/v1/occurrence/search?scientificName={species_name}&limit=300"
    res = requests.get(url)

    if res.status_code != 200:
        return pd.DataFrame()

    data = res.json()

    records = []

    for item in data.get("results", []):
        records.append({
            "species": item.get("species"),
            "country": item.get("country"),
            "year": item.get("year"),
            "lat": item.get("decimalLatitude"),
            "lon": item.get("decimalLongitude")
        })

    return pd.DataFrame(records)

df = load_data(selected_species)

if df.empty:
    st.error("No data found for this species.")
    st.stop()

df_clean = df.dropna(subset=["lat", "lon"])

# =========================
# FILTER BY COUNTRY
# =========================
countries = ["All"] + sorted(df_clean["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Filter by Country", countries)

if selected_country != "All":
    df_clean = df_clean[df_clean["country"] == selected_country]

# =========================
# 🔥 RISK INTENSITY MODEL (NEW)
# =========================
country_density = df_clean["country"].value_counts().to_dict()

df_clean["risk_score"] = df_clean["country"].map(country_density)

max_risk = df_clean["risk_score"].max()
df_clean["risk_norm"] = (df_clean["risk_score"] / max_risk) * 255

# =========================
# 🌍 MAP
# =========================
st.subheader(f"🌍 Global Distribution: {selected_species}")

map_fig = px.scatter_geo(
    df_clean,
    lat="lat",
    lon="lon",
    color="country",
    hover_name="species",
    projection="natural earth"
)

st.plotly_chart(map_fig, use_container_width=True)

# =========================
# 🔥 HEATMAP
# =========================
st.subheader("🔥 Biodiversity Hotspot Heatmap")

heat_fig = px.density_map(
    df_clean,
    lat="lat",
    lon="lon",
    radius=12,
    center=dict(lat=20, lon=0),
    zoom=0,
    map_style="carto-positron"
)

st.plotly_chart(heat_fig, use_container_width=True)

# =========================
# 📉 TREND
# =========================
st.subheader("📉 Observation Trend Over Time")

df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce")
trend = df_clean.groupby("year").size().reset_index(name="observations")

if len(trend) > 1:
    fig = px.line(trend, x="year", y="observations")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🧠 RISK INDICATOR
# =========================
st.subheader("🧠 Extinction Risk Indicator")

if len(trend) > 2:
    slope = (trend["observations"].iloc[-1] - trend["observations"].iloc[0]) / len(trend)

    if slope > 0:
        risk = "🟢 Stable / Increasing"
    elif slope > -2:
        risk = "🟡 Moderate Decline"
    else:
        risk = "🔴 High Decline Risk"

    st.metric("Risk Status", risk)

# =========================
# 🌎 COUNTRY STATS
# =========================
st.subheader("🌎 Country Distribution (Top 10)")

country_counts = df_clean["country"].value_counts().head(10)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(country_counts)

with col2:
    fig_pie = px.pie(values=country_counts.values, names=country_counts.index)
    st.plotly_chart(fig_pie)

# =========================
# 🧬 SPECIES COMPARISON
# =========================
if compare_species:

    comp_data = []

    for sp in compare_species:
        url = f"https://api.gbif.org/v1/occurrence/search?scientificName={sp}&limit=0"
        res = requests.get(url)

        if res.status_code == 200:
            total_count = res.json().get("count", 0)
        else:
            total_count = 0

        comp_data.append({
            "species": sp,
            "records": total_count
        })

    comp_df = pd.DataFrame(comp_data)

    st.subheader("🧬 Species Comparison")

    fig = px.bar(comp_df, x="species", y="records")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🧠 INSIGHTS
# =========================
st.subheader("🧠 Live Ecological Intelligence")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(df_clean))

with col2:
    st.metric("Countries Covered", df_clean["country"].nunique())

with col3:
    st.metric("Unique Coordinates", df_clean[["lat", "lon"]].drop_duplicates().shape[0])

with col4:
    score = len(df_clean) / max(df_clean["country"].nunique(), 1)
    st.metric("Data Spread Score", round(score, 2))

# =========================
# 🛰️ SATELLITE MODE
# =========================
st.subheader("🛰️ Satellite Mode: Earth Systems View")

sat_mode = st.selectbox(
    "Choose Satellite Layer",
    ["Night Lights (Human Activity)", "Vegetation Index (NDVI)"]
)

center_lat = df_clean["lat"].mean()
center_lon = df_clean["lon"].mean()

if sat_mode == "Night Lights (Human Activity)":

    layer = pdk.Layer(
        "TileLayer",
        data="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_CityLights_2012/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg",
        opacity=0.7
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=1
        )
    ))

elif sat_mode == "Vegetation Index (NDVI)":

    layer = pdk.Layer(
        "TileLayer",
        data="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg",
        opacity=0.7
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=1
        )
    ))

# =========================
# 🔴 RISK GLOW OVERLAY MAP (NEW)
# =========================
st.subheader("🔥 Ecological Risk Glow Map")

risk_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_clean,
    get_position='[lon, lat]',
    get_radius=50000,
    get_fill_color='[255, 0, 0, risk_norm]',
    opacity=0.6,
    pickable=True
)

glow_map = pdk.Deck(
    layers=[risk_layer],
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=1
    ),
    tooltip={"text": "Species: {species}\nCountry: {country}"}
)

st.pydeck_chart(glow_map)

st.info("🔥 Red glow shows ecological pressure zones (proxy model based on biodiversity density).")

# =========================
# FINAL INFO
# =========================
st.info("""
🌍 Live biodiversity intelligence system with:
- Distribution maps
- Heatmaps
- Trend analysis
- Risk estimation
- Species comparison
- Satellite Earth layers
- Ecological risk glow overlay
""")
