# app.py
import streamlit as st
import requests
import numpy as np
from skyfield.api import load, EarthSatellite
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set up the Streamlit page
st.set_page_config(
    page_title="Satellite Orbit Visualizer",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Interactive Satellite Orbit Visualizer")
st.markdown("""
This application fetches recent TLE data for 10 satellites and plots their orbits in 3D space. 
The visualization shows their paths over the **past 30 days**.
""")

# --- Part 1: Fetch TLEs for satellites ---
@st.cache_data(ttl=3600)  # Cache the data for 1 hour to avoid repeated requests
def fetch_tles():
    """Fetches TLE data for a list of NORAD IDs."""
    norad_ids = [28874, 25544, 25338, 858, 39199, 36112, 33401, 39197, 25560, 22824]
    tle_data = {}
    
    with st.spinner('Fetching satellite data from Celestrak...'):
        for norad in norad_ids:
            # The general purpose TLE link is the most reliable for current data.
            url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()  # Raise an HTTPError for bad responses
                tle_lines = resp.text.strip().split('\n')
                if len(tle_lines) == 3:
                    tle_data[norad] = tle_lines
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching TLE for NORAD ID {norad}: {e}")
                
    if not tle_data:
        st.error("Failed to fetch any satellite data. Please check your internet connection or try again later.")
        return None
        
    st.success(f"Successfully fetched TLEs for {len(tle_data)} satellites.")
    return tle_data

# --- Part 2: Compute satellite positions ---
@st.cache_data(ttl=3600) # Cache the position data
def compute_positions(tle_data):
    """Computes positions of satellites over a specified time span."""
    if not tle_data:
        return None

    # Time span: past 30 days
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
    with st.spinner('Calculating satellite positions... This may take a moment.'):
        for norad, tle_lines in tle_data.items():
            try:
                sat = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
                e = sat.at(sf_times)
                positions[norad] = (e.position.km[0], e.position.km[1], e.position.km[2])
            except Exception as e:
                st.warning(f"Could not compute position for {tle_lines[0]} (NORAD {norad}): {e}")
                continue
    
    st.success("Satellite positions computed successfully!")
    return positions, sf_times

# --- Part 3: Create and display the Plotly figure ---
def create_plot(positions, sf_times):
    """Generates the interactive 3D Plotly figure."""
    fig = go.Figure()

    colors = ['#FF4500', '#1E90FF', '#32CD32', '#FFD700', '#9400D3', '#00CED1', '#FF69B4', '#8B4513', '#696969', '#7CFC00']
    
    # === Draw Earth ===
    fig.add_trace(go.Mesh3d(
        x=[0], y=[0], z=[0], 
        alphahull=-1, 
        name='Earth'
    ))

    earth_radius = 6371  # Earth's radius in km
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    earth_x = earth_radius * np.cos(u) * np.sin(v)
    earth_y = earth_radius * np.sin(u) * np.sin(v)
    earth_z = earth_radius * np.cos(v)

    fig.add_trace(go.Surface(
        x=earth_x, y=earth_y, z=earth_z,
        colorscale='Earth', opacity=0.7, showscale=False,
        name='Earth'
    ))

    # === Initial Satellite Traces ===
    for idx, (norad, (x, y, z)) in enumerate(positions.items()):
        color = colors[idx % len(colors)]
        # Add full orbit path (static)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=color, width=1),
            name=f'Path NORAD {norad}',
            showlegend=False
        ))
        # Add starting point (first position)
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=5, color=color),
            name=f'NORAD {norad}'
        ))

    # === Create Animation Frames ===
    frames = []
    num_points = len(sf_times)
    
    animation_points = range(1, num_points)

    for t in animation_points:
        frame_traces = []
        for idx, (norad, (x, y, z)) in enumerate(positions.items()):
            color = colors[idx % len(colors)]
            frame_traces.append(go.Scatter3d(
                x=[x[t]], y=[y[t]], z=[z[t]],
                mode='markers',
                marker=dict(size=5, color=color),
                name=f'NORAD {norad}',
                showlegend=False
            ))
        frames.append(go.Frame(data=frame_traces, name=str(t)))
    
    fig.frames = frames

    # === Layout and Animation Buttons ===
    fig.update_layout(
        title="Satellite Orbits Over the Past 30 Days",
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="▶️ Play", method="animate",
                     args=[None, {
                         "frame": {"duration": 100, "redraw": True},
                         "fromcurrent": True
                     }]),
                dict(label="⏸️ Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}])
            ]
        )]
    )
    
    return fig

# --- Main App Logic ---
if __name__ == "__main__":
    tle_data = fetch_tles()
    if tle_data:
        positions, sf_times = compute_positions(tle_data)
        if positions:
            fig = create_plot(positions, sf_times)
            st.plotly_chart(fig, use_container_width=True)
