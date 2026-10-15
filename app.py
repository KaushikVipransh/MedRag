import streamlit as st

from medrag.config import CloudSettings

st.set_page_config(page_title="MedRAG", page_icon="⚕️", layout="wide")

if CloudSettings.from_env().backend == "local":
    from medrag.web import local as page
else:
    from medrag.web import multiuser as page

page.run()
