import streamlit as st

st.set_page_config(
    page_title="arXiv-GPT",
    page_icon="👋",
)

st.write("# Welcome to arXiv-GPT! 👋")

st.sidebar.success("Select a function above.")
st.sidebar.markdown("Current functions include visualizing papers in the arxiv embedding, or searching for similar papers to an input paper or prompt phrase.")

st.markdown(
    """
    arXiv+GPT is a framework for searching and visualizing papers on
    the [arXiv](https://arxiv.org/) using the context sensitivity from modern
    large language models (LLMs) like GPT3 to better link paper contexts
    
    **👈 Select a tool from the sidebar** to see some examples
    of what this framework can do!
    ### Want to learn more?
    - Check out `chaotic_neural` [(link)](http://chaotic-neural.readthedocs.io/)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Contribute!
"""
)
