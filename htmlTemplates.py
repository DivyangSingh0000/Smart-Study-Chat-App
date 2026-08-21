"""Shared visual styles for StudySmart."""

css = """
<style>
    .stApp { background: #90caf9; color: #0d47a1; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d47a1 0%, #2196f3 100%); }
    [data-testid="stSidebar"] * { color: #ffffff; }
    .accent { color: #2196f3; font-size: 0.7em; font-weight: 600; }
    .source-passage { background: #e3f2fd; border: 1px solid #2196f3; border-left: 5px solid #0d47a1; border-radius: 10px; color: #0d47a1; line-height: 1.7; padding: 1rem 1.2rem; }
    mark { background: #90caf9; color: #0d47a1; border-radius: 3px; padding: 0 .15rem; }
    [data-testid="stMetric"] { background: #e3f2fd; border: 1px solid #2196f3; border-radius: 12px; padding: .75rem; }
    [data-testid="stChatMessage"] { border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { background: #90caf9; border-radius: 10px; gap: .35rem; padding: .35rem; }
    .stTabs [data-baseweb="tab"] { background: #e3f2fd; border: 1px solid #2196f3; border-radius: 7px; color: #0d47a1; font-weight: 650; padding: .5rem .9rem; }
    .stTabs [data-baseweb="tab"]:hover { background: #ffffff; color: #0d47a1; }
    .stTabs [aria-selected="true"] { background: #0d47a1 !important; border-color: #0d47a1 !important; color: #ffffff !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stButton > button, .stDownloadButton > button { background: #0d47a1; border: 1px solid #0d47a1; border-radius: 7px; color: #ffffff; font-weight: 650; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #2196f3; border-color: #0d47a1; color: #ffffff; }
    .stButton > button:focus { box-shadow: 0 0 0 0.2rem #90caf9; }
</style>
"""
