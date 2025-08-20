import streamlit as st
import numpy as np
from skyfield.api import load, EarthSatellite
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Satellite Orbit Visualizer", page_icon="🛰️", layout="wide")

st.title("🛰️ Interactive Satellite Orbit Visualizer")
st.markdown("This app uses pre-saved TLE data (embedded inside the code) and plots orbits for the past 30 days.")

# --- Pre-saved TLE Data for 10 Satellites ---
tle_data = {
    "28874": ["USA 181",
              "1 28874U 05042A   24234.33562962  .00000046  00000-0  00000-0 0  9992",
              "2 28874   0.0173  88.9936 0001184  45.6725 314.4523  1.00271211 72243"],
    "25544": ["ISS (ZARYA)",
              "1 25544U 98067A   24234.51762566  .00013953  00000-0  25266-3 0  9995",
              "2 25544  51.6411  39.2543 0005177  94.4421 325.7228 15.49917990552855"],
    "25338": ["IRIDIUM 33",
              "1 25338U 98030A   24234.42050168  .00000134  00000-0  53715-4 0  9999",
              "2 25338  86.3983 211.1087 0002197  84.2379 275.9060 14.34215473285492"],
    "858": ["SEASAT 1",
            "1 00858U 76064A   24234.27037094  .00000063  00000-0  00000-0 0  9995",
            "2 00858  108.0033 322.6901 0003344 172.6571 187.4758 13.96037743249074"],
    "39199": ["COSMOS 2481",
              "1 39199U 13018A   24234.35673265  .00000078  00000-0  00000-0 0  9998",
              "2 39199  82.4761 268.6791 0013804 319.8972  40.1401 12.43015611517312"],
    "36112": ["COSMOS 2469",
              "1 36112U 09072A   24234.34890410 -.00000007  00000-0  00000-0 0  9992",
              "2 36112  67.1504 219.2729 0005559  92.7793 267.4201 12.87870826444837"],
    "33401": ["TERRA SAR-X",
              "1 33401U 08026A   24234.36498304  .00000012  00000-0  00000-0 0  9996",
              "2 33401  97.4471  41.6390 0001267 117.1127 242.9946 15.11966790748002"],
    "39197": ["COSMOS 2480",
              "1 39197U 13017A   24234.32517136  .00000066  00000-0  00000-0 0  9995",
              "2 39197  82.4803  89.0935 0014615  44.1765 316.0496 12.43010836517306"],
    "25560": ["NOAA 15",
              "1 25560U 98030A   24234.36247698  .00000093  00000-0  82172-4 0  9990",
              "2 25560  98.7105 238.2810 0011862  43.8911 316.3239 14.25943725285530"],
    "22824": ["COSMOS 2335",
              "1 22824U 93061A   24234.33950139  .00000049  00000-0  00000-0 0  9994",
              "2 22824  64.5568 126.3526 0012172  77.2059 283.0695 13.55836291497845"],
}

# --- Compute positions ---
@st.cache_data
def compute_positions(tle_data):
    time_step_hours = 6
    start_past = datetime.utcnow() - timedelta(days=30)
    end_past = datetime.utcnow()
    
    times = []
    current = start_past
    while current <= end_past:
        times.append(current)
        current += timedelta(hours=time_step_hours)
    
    ts = load.timescale()
    sf_times = ts.utc(
        np.array([t.year for t in times]),
        np.array([t.month for t in times]),
        np.array([t.day for t in times]),
        np.array([t.hour for t in times])
    )
    
    positions = {}
    for norad, tle_lines in tle_data.items():
        sat = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
        e = sat.at(sf_times)
        positions[norad] = (e.position.km[0], e.position.km[1], e.position.km[2])
    
    return positions, sf_times

positions, sf_times = compute_positions(tle_data)

# --- Plot Globe + Orbits ---
def create_plot(positions, sf_times):
    fig = go.Figure()
    colors = ['#FF4500','#1E90FF','#32CD32','#FFD700','#9400D3','#00CED1','#FF69B4','#8B4513','#696969','#7CFC00']
    
    # Earth sphere
    earth_radius = 6371
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    earth_x = earth_radius*np.cos(u)*np.sin(v)
    earth_y = earth_radius*np.sin(u)*np.sin(v)
    earth_z = earth_radius*np.cos(v)
    fig.add_trace(go.Surface(x=earth_x,y=earth_y,z=earth_z,colorscale='Earth',opacity=0.7,showscale=False,name='Earth'))
    
    # Satellite paths
    for idx, (norad,(x,y,z)) in enumerate(positions.items()):
        color=colors[idx%len(colors)]
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color=color,width=1),name=f'Path {norad}',showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x[0]],y=[y[0]],z=[z[0]],mode='markers',marker=dict(size=5,color=color),name=f'NORAD {norad}'))
    
    fig.update_layout(scene=dict(aspectmode='data'))
    return fig

fig = create_plot(positions, sf_times)
st.plotly_chart(fig, use_container_width=True)
