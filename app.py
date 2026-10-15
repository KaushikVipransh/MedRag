import os

import streamlit as st

from medrag.config import CloudSettings

st.set_page_config(page_title="MedRAG", page_icon="⚕️", layout="wide")

# Streamlit Community Cloud supplies settings as st.secrets; the rest of MedRAG reads the
# environment. Real environment variables win.
try:
    for name, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(name, str(value))
except Exception:  # no secrets configured
    pass

if CloudSettings.from_env().backend == "local":
    from medrag.web import local as page
else:
    from medrag.web import multiuser as page

page.run()
