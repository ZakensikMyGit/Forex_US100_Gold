"""Panel analizy technicznej: US100 i złoto.

Uruchomienie:  .\\uruchom.ps1   (albo: .venv\\Scripts\\python -m streamlit run streamlit_app.py)
"""

import streamlit as st

st.set_page_config(page_title="Panel analizy – US100 i złoto", page_icon=":material/candlestick_chart:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/us100.py", title="US100", icon=":material/candlestick_chart:", default=True),
        st.Page("app_pages/zloto.py", title="Złoto", icon=":material/diamond:", url_path="zloto"),
    ],
    position="top",
)
page.run()
