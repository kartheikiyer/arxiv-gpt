import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from bokeh.palettes import OrRd
from bokeh.plotting import figure, show
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
import cloudpickle as cp
from scipy import stats
from urllib.request import urlopen

st.title("ArXiv+GPT3 embedding explorer")
st.markdown("This is an explorer for astro-ph.GA papers on the arXiv (up to Apt 18th, 2023). The papers have been preprocessed with `chaotic_neural` [(link)](http://chaotic-neural.readthedocs.io/) after which the collected abstracts are run through `text-embedding-ada-002` with [langchain](https://python.langchain.com/en/latest/ecosystem/openai.html) to generate a unique vector correpsonding to each paper. These are then compressed using umap and shown here, and can be used for similarity searches with methods like [faiss](https://github.com/facebookresearch/faiss). The scatterplot here can be paired with a heatmap for more targeted searches looking at a specific topic or area (see sidebar). Upgrade to chaotic neural suggested by Jo Ciuca, thank you! More to come (hopefully) with GPT-4 and its applications!")

@st.cache_data
def get_embedding_data(url):
    data = cp.load(urlopen(url))
    st.sidebar.success("Fetched data from API!")
    return data

url = "https://drive.google.com/uc?export=download&id=1133tynMwsfdR1wxbkFLhbES3FwDWTPjP"
embedding, all_text, all_titles, all_arxivid, all_links = get_embedding_data(url)

def density_estimation(m1, m2, xmin=0, ymin=0, xmax=15, ymax=15):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z
        
st.sidebar.markdown('This is a widget that allows you to look for papers containing specific phrases in the dataset and show it as a heatmap. Enter the phrase of interest, then change the size and opacity of the heatmap as desired to find the high-density regions. Hover over blue points to see the details of individual papers.')
    
st.sidebar.text_input("Search query", key="phrase", value="")
alpha_value = st.sidebar.slider("Pick the hexbin opacity",0.0,1.0,0.1)
size_value = st.sidebar.slider("Pick the hexbin size",0.0,2.0,0.2)

phrase=st.session_state.phrase

phrase_flags = np.zeros((len(all_text),))
for i in range(len(all_text)):
    if phrase.lower() in all_text[i].lower():
        phrase_flags[i] = 1
        

source = ColumnDataSource(data=dict(
    x=embedding[0:,0],
    y=embedding[0:,1],
    title=all_titles,
    link=all_links,
))

TOOLTIPS = """
<div style="width:300px;">
ID: $index
($x, $y)
@title <br>
@link <br> <br>
</div>
"""
        
p = figure(width=700, height=583, tooltips=TOOLTIPS, x_range=(0, 15), y_range=(2.5,15),
           title="UMAP projection of trained ArXiv corpus | heatmap keyword: "+phrase)

p.hexbin(embedding[phrase_flags==1,0],embedding[phrase_flags==1,1], size=size_value, 
         palette = np.flip(OrRd[8]), alpha=alpha_value)
p.circle('x', 'y', size=3, source=source, alpha=0.3)

st.bokeh_chart(p)