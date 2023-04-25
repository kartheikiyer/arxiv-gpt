import datetime, os
import chaotic_neural as cn
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai
import faiss
import streamlit as st
import feedparser
import urllib
import cloudpickle as cp
from urllib.request import urlopen
from summa import summarizer

# openai.organization = "org-EBvNTjd2pLrK4vOhFlNMLr2v"
openai.organization = st.secrets.openai.org
openai.api_key = st.secrets.openai.api_key
os.environ["OPENAI_API_KEY"] = openai.api_key

@st.cache_data
def get_feeds_data(url):
    data = cp.load(urlopen(url))
    st.sidebar.success("Fetched data from API!")
    return data

embeddings = OpenAIEmbeddings()

feeds_link = "https://drive.google.com/uc?export=download&id=1-IPk1voyUM9VqnghwyVrM1dY6rFnn1S_"
embed_link = "https://dl.dropboxusercontent.com/s/ob2betm29qrtb8v/astro_ph_ga_feeds_ada_embedding_18-Apr-2023.pkl?dl=0"
gal_feeds = get_feeds_data(feeds_link)
arxiv_ada_embeddings = get_feeds_data(embed_link)

ctr = -1
num_chunks = len(gal_feeds)
all_text, all_titles, all_arxivid, all_links, all_authors = [], [], [], [], []

for nc in tqdm(range(num_chunks)):

    for i in range(len(gal_feeds[nc].entries)):
        text = gal_feeds[nc].entries[i].summary
        text = text.replace('\n', ' ')
        text = text.replace('\\', '')
        all_text.append(text)
        all_titles.append(gal_feeds[nc].entries[i].title)
        all_arxivid.append(gal_feeds[nc].entries[i].id.split('/')[-1][0:-2])
        all_links.append(gal_feeds[nc].entries[i].links[1].href)
        all_authors.append(gal_feeds[nc].entries[i].authors)

d = arxiv_ada_embeddings.shape[1]                           # dimension
nb = arxiv_ada_embeddings.shape[0]                      # database size
xb = arxiv_ada_embeddings.astype('float32')
index = faiss.IndexFlatL2(d)
index.add(xb)

def run_simple_query(search_query = 'all:sed+fitting', max_results = 10, start = 0, sort_by = 'lastUpdatedDate', sort_order = 'descending'):
    """
        Query ArXiv to return search results for a particular query
        Parameters
        ----------
        query: str
            query term. use prefixes ti, au, abs, co, jr, cat, m, id, all as applicable.
        max_results: int, default = 10
            number of results to return. numbers > 1000 generally lead to timeouts
        start: int, default = 0
            start index for results reported. use this if you're interested in running chunks.
        Returns
        -------
        feed: dict
            object containing requested results parsed with feedparser
        Notes
        -----
            add functionality for chunk parsing, as well as storage and retreival
        """

    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?';
    query = 'search_query=%s&start=%i&max_results=%i&sortBy=%s&sortOrder=%s' % (search_query,
                                                     start,
                                                     max_results,sort_by,sort_order)

    response = urllib.request.urlopen(base_url+query).read()
    feed = feedparser.parse(response)
    return feed

def find_papers_by_author(auth_name):

    doc_ids = []
    for doc_id in range(len(all_authors)):
        for auth_id in range(len(all_authors[doc_id])):
            if auth_name.lower() in all_authors[doc_id][auth_id]['name'].lower():
                print('Doc ID: ',doc_id, ' | arXiv: ', all_arxivid[doc_id], '| ', all_titles[doc_id],' | Author entry: ', all_authors[doc_id][auth_id]['name'])
                doc_ids.append(doc_id)

    return doc_ids

def faiss_based_indices(input_vector, nindex=10):
    xq = input_vector.reshape(-1,1).T.astype('float32')
    D, I = index.search(xq, nindex)
    return I[0], D[0]


def list_similar_papers_v2(model_data,
                        doc_id = [], input_type = 'doc_id',
                        show_authors = False, show_summary = False,
                        return_n = 10):

    arxiv_ada_embeddings, embeddings, all_titles, all_abstracts, all_authors = model_data

    if input_type == 'doc_id':
        print('Doc ID: ',doc_id,', title: ',all_titles[doc_id])
#         inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        inferred_vector = arxiv_ada_embeddings[doc_id,0:]
        start_range = 1
    elif input_type == 'arxiv_id':
        print('ArXiv id: ',doc_id)
        arxiv_query_feed = run_simple_query(search_query='id:'+str(doc_id))
        if len(arxiv_query_feed.entries) == 0:
            print('error: arxiv id not found.')
            return
        else:
            print('Title: '+arxiv_query_feed.entries[0].title)
            inferred_vector = cn.np.array(embeddings.embed_query(arxiv_query_feed.entries[0].summary))
#         arxiv_query_tokens = gensim.utils.simple_preprocess(arxiv_query_feed.entries[0].summary)
#         inferred_vector = model.infer_vector(arxiv_query_tokens)

        start_range = 0
    elif input_type == 'keywords':
#         print('Keyword(s): ',[doc_id[i] for i in range(len(doc_id))])
#         word_vector = model.wv[doc_id[0]]
#         if len(doc_id) > 1:
#            print('multi-keyword')
#            for i in range(1,len(doc_id)):
#                word_vector = word_vector + model.wv[doc_id[i]]
# #         word_vector = model.infer_vector(doc_id)
#         inferred_vector = word_vector
        inferred_vector = cn.np.array(embeddings.embed_query(doc_id))
        start_range = 0
    else:
        print('unrecognized input type.')
        return

#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    sims, dists = faiss_based_indices(inferred_vector, return_n+2)
    textstr = ''

    textstr = textstr + '-----------------------------\n'
    textstr = textstr + 'Most similar/relevant papers: \n'
    textstr = textstr + '-----------------------------\n\n'
    for i in range(start_range,start_range+return_n):

        # print(i, all_titles[sims[i]], ' (Distance: %.2f' %dists[i] ,')')
        textstr = textstr + str(i+1)+'. **'+ all_titles[sims[i]] +'** (Distance: %.2f' %dists[i]+')   \n'
        textstr = textstr + '**ArXiv:** ['+all_arxivid[sims[i]]+'](https://arxiv.org/abs/'+all_arxivid[sims[i]]+')  \n'
        if show_authors == True:
            textstr = textstr + '**Authors:**  '
            temp = all_authors[sims[i]]
            for ak in range(len(temp)):
                if ak < len(temp)-1:
                    textstr = textstr + temp[ak].name + ', '
                else:
                    textstr = textstr + temp[ak].name + '   \n'
        if show_summary == True:
            textstr = textstr + '**Summary:**  '
            text = all_text[sims[i]]
            text = text.replace('\n', ' ')
            textstr = textstr + summarizer.summarize(text) + '  \n'
        if show_authors == True or show_summary == True:
            textstr = textstr + ' '
        textstr = textstr + '  \n'
    return textstr


model_data = [arxiv_ada_embeddings, embeddings, all_titles, all_text, all_authors]

st.title('ArXiv similarity search:')
st.markdown('Search for similar papers by arxiv id or phrase:')

search_type = st.radio(
    "What are you searching by?",
    ('arxiv id', 'text query'), index=1)

query = st.text_input('Search query or arxivid', value="what causes galaxy quenching?")
show_authors = st.checkbox('Show author information', value = True)
show_summary = st.checkbox('Show paper summary', value = True)
return_n = st.slider('How many papers should I show?', 1, 30, 10)

if search_type == 'arxiv id':
    sims = list_similar_papers_v2(model_data, doc_id = query, input_type='arxiv_id', show_authors = show_authors, show_summary = show_summary, return_n = return_n)
else:
    sims = list_similar_papers_v2(model_data, doc_id = query, input_type='keywords', show_authors = show_authors, show_summary = show_summary, return_n = return_n)

st.markdown(sims)
