"""Shared design tokens and CSS for the app and the developer evaluation page."""
import html

import streamlit as st

# ---- design tokens ---------------------------------------------------------------------------------
MIST, PAPER, INK, SOFT, LINE = "#EEF1F5", "#FFFFFF", "#1C2433", "#5B6577", "#DCE2EA"
SLATE, SLATE_TINT, SAGE, SAGE_TINT = "#3B5B7A", "#E3EAF2", "#55796A", "#EAF1ED"
ROSE, ROSE_TINT = "#9E3650", "#FBEEF1"
CHART = ["#3B5B7A", "#8FA9C2", "#55796A", "#A9C4B7", "#9E3650", "#C9CED6"]
esc = html.escape

CSS = f"""
<style>
@font-face {{ font-family:'Readex Pro'; src:url('app/static/ReadexPro.ttf') format('truetype'); font-weight:160 700; font-display:swap; }}
@font-face {{ font-family:'Noto Naskh Arabic'; src:url('app/static/NotoNaskhArabic.ttf') format('truetype'); font-weight:400 700; font-display:swap; }}
:root {{ --ui: 'Readex Pro', system-ui, sans-serif; --read: 'Noto Naskh Arabic', serif; }}
html, body, .stApp, button, input, textarea, select, label, p, div, span {{ font-family: var(--ui) !important; }}
.stApp {{ background:{MIST}; color:{INK}; }}
.block-container {{ direction:rtl; text-align:right; max-width:980px; padding:2.4rem 1.5rem 4rem; }}
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stDecoration"], [data-testid="stElementToolbar"],
[data-testid="stStatusWidget"], [data-testid="stIconMaterial"] {{ display:none !important; }}
*:focus-visible {{ outline:2px solid {SLATE} !important; outline-offset:2px; }}

.top {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem; flex-wrap:wrap; }}
.name {{ font-size:2.6rem; font-weight:600; color:{INK}; line-height:1.15; }}
.tagline {{ color:{SOFT}; font-size:1rem; margin-top:.35rem; }}
.state {{ font-size:.85rem; color:{SOFT}; line-height:1.8; }}
.state b {{ color:{INK}; font-weight:500; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-left:.4rem; vertical-align:middle; }}

.stTabs [data-baseweb="tab-list"] {{ gap:2rem; border-bottom:1px solid {LINE}; direction:rtl; margin-top:1.2rem; }}
.stTabs [data-baseweb="tab"] {{ padding:.75rem 0; font-size:.98rem; color:{SOFT}; background:transparent; }}
.stTabs [aria-selected="true"] {{ color:{INK}; font-weight:500; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{SLATE}; height:2px; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:1.4rem; }}

.welcome {{ text-align:center; padding:2.2rem 1rem 1.4rem; }}
.welcome h2 {{ font-family:var(--read) !important; font-size:2.2rem; font-weight:500; color:{INK}; margin:0 0 .7rem; }}
.welcome p {{ color:{SOFT}; max-width:560px; margin:0 auto; line-height:1.95; font-size:1rem; }}

.you {{ background:{SLATE_TINT}; color:{INK}; border-radius:18px 18px 4px 18px; padding:.85rem 1.15rem;
        margin:1.4rem 0 .7rem 0; margin-left:auto; max-width:78%; width:fit-content; line-height:1.9; }}
.reply {{ background:{PAPER}; border:1px solid {LINE}; border-radius:18px; padding:1.5rem 1.7rem 1.2rem; max-width:92%; }}
.reply .say {{ font-family:var(--read) !important; font-size:1.18rem; line-height:2.05; color:{INK}; }}
.reply .label {{ color:{SOFT}; font-size:.85rem; margin-bottom:.35rem; }}
.reply ol {{ margin:.9rem 1.4rem 0 0; padding:0; direction:rtl; text-align:right; }}
.reply li {{ direction:rtl; text-align:right; padding-right:.3rem; font-family:var(--read) !important; font-size:1.05rem; line-height:1.95; margin-bottom:.2rem; }}
.meta {{ display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; margin-top:1.1rem; color:{SOFT}; font-size:.85rem; }}
.tag {{ background:{SLATE_TINT}; color:{SLATE}; border-radius:999px; padding:.15rem .75rem; font-size:.82rem; }}
.record {{ margin-top:1.3rem; border-top:1px solid {LINE}; padding-top:1rem; }}
.record .title {{ font-size:.92rem; font-weight:500; color:{INK}; margin:0 0 .7rem; }}
.case {{ border-right:3px solid {SAGE}; background:{SAGE_TINT}; padding:.75rem 1rem; margin-bottom:.6rem; }}
.case .q {{ color:{SOFT}; font-size:.88rem; line-height:1.8; }}
.case .a {{ font-family:var(--read) !important; color:#26392F; font-size:1rem; line-height:1.95; margin-top:.25rem; }}
.gpt-off {{ margin-top:.8rem; padding:.6rem .9rem; background:#F4F6F9; border:1px dashed {LINE}; border-radius:10px;
            color:{SOFT}; font-size:.85rem; line-height:1.8; }}
.fine {{ color:{SOFT}; font-size:.84rem; line-height:1.8; margin-top:1rem; }}

.crisis {{ background:{ROSE_TINT}; border:1px solid #EFCBD4; border-radius:18px; padding:1.6rem 1.7rem; max-width:92%; }}
.crisis .say {{ font-family:var(--read) !important; font-size:1.18rem; line-height:2.05; color:#4A1626; }}
.lines {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px,1fr)); gap:.7rem; margin-top:1.2rem; }}
.line {{ background:{PAPER}; border:1px solid #EFCBD4; border-radius:12px; padding:.8rem 1rem; }}
.line b {{ display:block; font-size:1.55rem; font-weight:600; color:{ROSE}; }}
.line span {{ font-size:.85rem; color:#5B2A36; line-height:1.7; }}

textarea {{ direction:rtl; text-align:right; font-size:1rem !important; line-height:1.8 !important; }}
[data-testid="stTextArea"] > div > div {{ border-radius:14px !important; }}
.stButton > button {{ border-radius:999px; border:1px solid {LINE}; background:{PAPER}; color:{INK}; font-weight:400;
                      padding:.5rem 1.1rem; white-space:normal; line-height:1.6; }}
.stButton > button p {{ white-space:normal !important; overflow:visible !important; text-overflow:clip !important; text-align:center; }}
.stButton > button:hover {{ border-color:{SLATE}; color:{SLATE}; }}
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button p,
.stButton > button[kind="primary"], .stButton > button[kind="primary"] p {{
  background:{SLATE} !important; border-color:{SLATE} !important; color:#fff !important; border-radius:12px !important; }}
.help-strip {{ text-align:center; color:{SOFT}; font-size:.88rem; margin-top:1.6rem; line-height:1.9; }}
.help-strip b {{ color:{ROSE}; font-weight:600; }}
.notice {{ background:{PAPER}; border:1px solid {LINE}; border-right:3px solid {SLATE};
           padding:.8rem 1.1rem; color:{SOFT}; font-size:.9rem; line-height:1.8; margin:1rem 0; }}

.sec {{ display:flex; gap:1rem; align-items:baseline; margin:2.4rem 0 .3rem; }}
.sec .n {{ font-size:1.9rem; font-weight:300; color:{SLATE}; min-width:1.6rem; }}
.sec h3 {{ font-size:1.25rem; font-weight:500; color:{INK}; margin:0; }}
.lead {{ color:{SOFT}; font-size:.95rem; line-height:1.9; margin:0 2.6rem 1rem 0; max-width:720px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px,1fr)); gap:.8rem; margin-bottom:.8rem; }}
.stat {{ background:{PAPER}; border:1px solid {LINE}; border-radius:14px; padding:1rem 1.1rem; }}
.stat b {{ display:block; font-size:1.7rem; font-weight:500; color:{INK}; font-variant-numeric:tabular-nums; }}
.stat span {{ color:{INK}; font-size:.86rem; line-height:1.6; }}
.stat i {{ display:block; font-style:normal; color:{SOFT}; font-size:.78rem; margin-top:.3rem; line-height:1.6; }}
.meter {{ height:4px; background:{LINE}; border-radius:2px; margin-top:.6rem; overflow:hidden; }}
.meter span {{ display:block; height:100%; background:{SLATE}; }}
.honest {{ background:{PAPER}; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.3rem; line-height:1.95;
           color:{INK}; margin-top:1.6rem; }}
.deflist {{ width:100%; border-collapse:collapse; background:{PAPER}; border:1px solid {LINE}; }}
.deflist td {{ padding:.75rem 1.1rem; border-bottom:1px solid {LINE}; vertical-align:top; line-height:1.85; font-size:.94rem; }}
.deflist tr:last-child td {{ border-bottom:none; }}
.deflist td:first-child {{ width:28%; color:{INK}; font-weight:500; }}
.deflist td:last-child {{ color:{SOFT}; }}
.prose {{ font-family:var(--read) !important; font-size:1.12rem; line-height:2.05; color:{INK}; max-width:760px; }}
.footer {{ margin-top:3rem; padding-top:1.1rem; border-top:1px solid {LINE}; color:{SOFT}; font-size:.85rem;
           line-height:1.8; text-align:center; }}
@media (max-width:700px) {{ .you, .reply, .crisis {{ max-width:100%; }} .name {{ font-size:2.1rem; }} .lead {{ margin-right:0; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
"""



def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
