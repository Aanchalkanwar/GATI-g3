import streamlit as st
import json
import numpy as np
from skyfield.api import load, EarthSatellite
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Satellite Orbit Visualizer", page_icon="🛰️", layout="wide")

st.title("🛰️ Interactive Satellite Orbit Visualizer")
st.markdown("This app loads saved TLE data (from file) and plots orbits for the past 30 days.")

# --- Load TLE Data ---
@st.cache_data
def load_tle_file():
    with open("tle_data.json", "r") as f:
        tle_data = json.load(f)
    return tle_data

tle_data = load_tle_file()

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

# --- Plot (same function as before) ---
def create_plot(positions, sf_times):
    fig = go.Figure()
    colors = ['#FF4500','#1E90FF','#32CD32','#FFD700','#9400D3','#00CED1','#FF69B4','#8B4513','#696969','#7CFC00']
    earth_radius = 6371
    u,v=np.mgrid[0:2*np.pi:50j,0:np.pi:25j]
    earth_x=earth_radius*np.cos(u)*np.sin(v)
    earth_y=earth_radius*np.sin(u)*np.sin(v)
    earth_z=earth_radius*np.cos(v)

    fig.add_trace(go.Surface(x=earth_x,y=earth_y,z=earth_z,colorscale='Earth',opacity=0.7,showscale=False,name='Earth'))
    
    for idx,(norad,(x,y,z)) in enumerate(positions.items()):
        color=colors[idx%len(colors)]
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color=color,width=1),name=f'Path {norad}',showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x[0]],y=[y[0]],z=[z[0]],mode='markers',marker=dict(size=5,color=color),name=f'NORAD {norad}'))
    
    return fig

fig = create_plot(positions, sf_times)
st.plotly_chart(fig, use_container_width=True)
