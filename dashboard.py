# dashboard.py: political subreddits analysis dashboard
# Cassandra M. @ 2025-04-09
# Jason L., Danielle T., Anna J., Mahin C.
# Grp. 2 @ BDC800NAA, Final Presentation

import streamlit as st
from PIL import Image
import os
import requests
from io import BytesIO
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# NOTE: More Streamlit documentation and methods found in: https://docs.streamlit.io/get-started/fundamentals/main-concepts

# Constants
MAIN_TITLE = "Reddit Political Analysis Dashboard"
# SUBREDDITS = ["r/politics", "r/liberals", "r/communist", "r/conservative", 
#              "r/canada", "r/usa", "r/trump", "r/fascist", "r/opinions",
#              "r/worldnews", "r/news", "r/democrats"]
SUBREDDITS = ["politics", "news", "democrats", "conservative", "canada", "worldnews"]
REDDIT_LOGO = "reddit_favicon.jpeg"
DATA_DIR = Path("data/processed")
CACHE_EXPIRATION = 3600  # 1 hour

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data layer
class DataManager:
    """Handle all data operations with caching and fallback"""

    def __init__(self):
        self._validate_data_structure()

    def _validate_data_structure(self) -> None:
        """Ensure required data files exist"""
        required_files = {f"{sub}/metadata.json" for sub in SUBREDDITS}
        missing_files = [f for f in required_files if not (DATA_DIR / f).exists()]
        if missing_files:
            logger.warning(f"Missing data files: {missing_files}")

    @st.cache_data(ttl=CACHE_EXPIRATION, show_spinner=False)
    def get_subreddit_data(_self, subreddit: str) -> Dict:
        """
        Get subreddit data with caching

        Args:
            subreddit: Name of subreddit (no '/r' prefix)
        Returns:
            dict: containing all analysis data
        """
        try:
            with open(DATA_DIR / f"{subreddit}/metadata.json") as f:
                data = json.load(f)

            # Add metadata
            data["metadata"] = {
                "source": f"preprocessed/{subreddit}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subreddit": subreddit
            }
            return data
        
        except Exception as e:
            logger.error(f"Error loading r/{subreddit} data: {str(e)}")
            return {}

    @staticmethod
    @st.cache_data(ttl=CACHE_EXPIRATION)
    def load_img(url: str) -> Optional[Image.Image]:
        """Cached image loader"""
        try:
            response = requests.get(url, timeout=10)
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Image load failed: {str(e)}")
            return None

# Visualization components
def create_metric_card(title: str, value, delta: str = None) -> str:
    """
    Creates a styled metric card with dynamic colouring

    Args:
        title: Metric name
        value: Primary value to display
        delta: Optional change indicator
    Returns:
        HTML string for the card
    """
    color_map = {
        "toxicity": ("#ff4b4b", "#ffdfdf"),
        "average": ("#00bfa0", "#e6fffa"),
        "activity": ("#00b1ee", "#c1efff"),
        "default": ("#f0f2f6", "#ffffff")
    }
    bg_color = color_map.get(title.split()[0].lower(), color_map["default"])

    return f"""
    <div style="
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, {bg_color[0]} 0%, {bg_color[1]} 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 16px;
    ">
        <h3 style="margin-top:0;color:#333;">{title}</h3>
        <h2 style="margin-bottom:0;color:#111;">{value}</h2>
        {f'<p style="color:#666;font-size:14px;">{delta}</p>' if delta else ''}
    </div>
    """

def render_political_lean(data: Dict) -> go.Figure:
    """
    Interactive political leanings visualization

    Args:
        data: Subreddit analysis data
    Returns:
        Plotly Figure object
    """
    lean_data = data["political_lean"]

    # Calculate center percentage (i.e., what's not left or right)
    center_pct = max(0, 1 - (lean_data["left"] + lean_data["right"]))
    
    fig = go.Figure()

    # Add bars for each leaning
    fig.add_trace(go.Bar(
        x=["Left", "Centre", "Right"],
        y=[lean_data["right"], center_pct, lean_data["left"]],
        marker_color=["#1f77b4", "#2ca02c", "#d62728"],
        text=[f"{lean_data['left']:.1%}", f"{center_pct:.1%}", f"{lean_data['right']:.1%}"],
        textposition="auto"
    ))

    # Add predicted lean indicator
    predicted = "Left" if lean_data["right"] > lean_data["left"] else "Right"
    fig.add_annotation(
        x=predicted,
        y=max(lean_data["left"], lean_data["right"]) + 0.05,
        text=f"Predicted: {predicted}",
        showarrow=False,
        font=dict(size=14, color="gray")
    )

    fig.update_layout(
        title="Political Lean Distribution",
        yaxis_title="Percentage (%) of Comments",
        yaxis_tickformat=".0%",
        xaxis_title="Political Orientation",
        template="plotly_white",
        height=500
    )
    return fig

def create_lean_metric(data: Dict) -> str:
    """
    Special metric card for political leanings
    """
    lean = data["political_lean"]
    bias_score = lean["right"] - lean["left"]
    predicted = "Left" if lean["right"] > lean["left"] else "Right"
    color_map = {
        "bias": ("#6200ee", "#f3e5ff"),
        "default": ("#f0f2f6", "#ffffff")
    }
    bg_color = color_map.get("bias", color_map["default"])

    return f"""
    <div style="
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, {bg_color[0]} 0%, {bg_color[1]} 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 16px;
    ">
        <h3 style="margin-top:0;color:#333;">Political Bias 🧭</h3>
        <h2 style="margin-bottom:0;color:#111;">{predicted}</h2>
        <p style="color:#666;font-size:14px;">
            {abs(bias_score):.1%} { 'More left' if bias_score > 0 else 'More right'}-leaning
        </p>
    </div>
    """

def render_toxicity_analysis(data: Dict) -> Tuple[go.Figure, go.Figure]:
    """
    Generate toxicity-related visualizations

    Args:
        data: Subreddit analysis data
    Returns:
        Tuple of (toxicity_distribution_fig, toxicity_sentiment_fig)
    """
    # Toxicity Distribution Pie Chart
    toxic_dist = {
        "Toxic": data["toxic_comments_ratio"],
        "Non-Toxic": 1 - data["toxic_comments_ratio"]
    }

    dist_fig = px.pie(
        names=list(toxic_dist.keys()),
        values=list(toxic_dist.values()),
        title=f"Toxicity Distribution (Average: {data['toxicity_avg']:.2f})",
        color_discrete_sequence=["#d3d3d3", "#ff4b4b"],
        hole=0.3
    )
    dist_fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{percent}</br>"
    )

    # Toxicity vs Sentiment Relationship
    # Create normalized comparison
    tox_norm = data["toxicity_avg"]  # Already 0-1 scaled
    sent_norm = (data["sentiment_avg"] + 1) / 2  # Convert -1 to +1 into 0-1 scale
    
    fig = go.Figure()
    
    # Add bars with explicit scales
    fig.add_trace(go.Bar(
        x=["Toxicity"],
        y=[tox_norm],
        name="Toxicity",
        marker_color="#ff4b4b",
        text=[f"Avg: {data['toxicity_avg']:.2f}"],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=["Sentiment"],
        y=[sent_norm],
        name="Sentiment",
        marker_color="#00bfa0",
        text=[f"Avg: {data['sentiment_avg']:.2f}"],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Normalized Comparison (0-1 Scale)",
        yaxis_title="Normalized Score",
        yaxis_range=[0, 1],
        barmode='group',
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Higher values = More Toxic/Positive",
                showarrow=False
            )
        ]
    )
    
    return dist_fig, fig

# User interface (UI) components
def setup_page() -> None:
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=MAIN_TITLE,
        page_icon=REDDIT_LOGO,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .stSelectbox div[data-baseweb="select"] { border-radius: 8px; }
        .stButton>button { border-radius: 8px; padding: 8px 16px; }
        .css-1aumxhk { background-color: #f0f2f6; }
        .header { color: #FF5700; }
        .metric-label { font-size: 1rem; color: #666; }
        .metric-value { font-size: 1.5rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar(dm: DataManager) -> str:
    """Render sidebar and return selected subreddit"""
    with st.sidebar:
        st.image(REDDIT_LOGO, width=80)
        st.title("Reddit Analyzer")
        st.markdown("---")

        subreddit = st.selectbox(
            "Select Subreddit",
            options=SUBREDDITS,
            format_func=lambda x: f"r/{x}",
            key="subreddit_select"
        )

        if st.button("Analyze", key="analyze_btn"):
            st.session_state.selected_subreddit = subreddit
            st.rerun()

        st.markdown("---")
        st.markdown("""
        **Metrics Guide:**
        - 🚩 Toxicity: 0-1 scale (higher = more toxic)
        - 🧭 Bias: difference of higher orientation
        - 😁 Sentiment: -1 (negative) to +1 (positive)
        """)

    return st.session_state.get("selected_subreddit", SUBREDDITS[0])

# Main app
def main():
    """Main application flow"""
    setup_page()
    dm = DataManager()

    # Sidebar and data loading
    selected_subreddit = render_sidebar(dm)
    data = dm.get_subreddit_data(selected_subreddit)

    if not data:
        st.error("Failed to load subreddit data")
        return
    
    # Main content area
    st.title(f"r/{selected_subreddit}: Political Analysis 🧑‍⚖️")
    st.markdown("---")

    # Key Metrics Row
    with st.container():
        cols = st.columns(4)
        with cols[0]:
            st.markdown(create_metric_card(
                "Toxicity Score 🚩",
                f"{data['toxicity_avg']:.2f}",
                f"{data['toxic_comments_ratio']:.1%} toxic comments"
            ), unsafe_allow_html=True)

        with cols[1]:
            st.markdown(create_lean_metric(data), unsafe_allow_html=True)

        with cols[2]:
            st.markdown(create_metric_card(
                "Average Sentiment 😁",
                f"{data['sentiment_avg']:.2f}",
                f"{data['sentiment_distribution']['positive']:.1%} positive (+)"
            ), unsafe_allow_html=True)

        with cols[3]:
            st.markdown(create_metric_card(
                "Activity Level 📊",
                f"{data['total_comments']:,} comments",
                f"{data['avg_upvotes']:.0f} average upvotes"
            ), unsafe_allow_html=True)

    # Visualizations
    with st.container():
        tab1, tab2, tab3 = st.tabs(["Political Leanings", "Toxicity Trends", "Word Cloud"])

        with tab1:
            st.plotly_chart(
                render_political_lean(data),
                use_container_width=True
            )

            # Sentiment distribution pie chart
            fig = px.pie(
                names=list(data["sentiment_distribution"].keys()),
                values=list(data["sentiment_distribution"].values()),
                title="Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            dist_fig, rel_fig = render_toxicity_analysis(data)

            st.plotly_chart(
                dist_fig,
                use_container_width=True
            )

            st.plotly_chart(
                rel_fig,
                use_container_width=True
            )

            # Add toxic phrases
            if "top_toxic_phrases" in data:
                st.subheader("Top Toxic Phrases")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Most Frequent:**")
                    for phrase in data["top_toxic_phrases"]["frequent"][:5]:
                        st.write(f"- {phrase}")
                with cols[1]:
                    st.markdown("**Most Severe:**")
                    for phrase in data["top_toxic_phrases"]["severe"][:5]:
                        st.write(f"- {phrase}")

        with tab3:
            col1 = st.columns(1)
            wc_left_path = DATA_DIR / f"{selected_subreddit}/wordcloud_left.png"
            wc_right_path = DATA_DIR / f"{selected_subreddit}/wordcloud_right.png"

            # Word Clouds
            with col1[0]:
                if wc_left_path.exists():
                    try:
                        img = Image.open(wc_left_path)
                        st.image(img, caption="Word Cloud")
                    except Exception as e:
                        st.error(f"Failed to load word cloud: {e}")
                elif wc_right_path.exists():
                    try:
                        img = Image.open(wc_right_path)
                        st.image(img, caption="Word Cloud")
                    except Exception as e:
                        st.error(f"Failed to load word cloud: {e}")

    # Footer
    st.markdown("---")
    st.caption(f"Last Updated: {data['metadata']['timestamp']} | Data Source: {data['metadata']['source']}")
    st.caption(f"BDC800 Capstone Project - Group 2 (Cassandra M., Jason L., Anna J., Danielle T., Mahin C.)")

if __name__ == "__main__":
    main()