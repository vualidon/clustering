from datasets import load_dataset
import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import plotly.express as px

import faiss
import numpy as np
import pandas as pd
# import plotly.express as px
from huggingface_hub import InferenceClient
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP
import google.generativeai as genai


summary_template = """{}\n\nUse three words total (comma separated)\
to describe general topics in above texts. Under no circumstances use enumeration. \
Example format: Tree, Cat, Fireman"""
# summary_template = """{}\n\nUse one word\
# to describe general topics in above texts. Under no circumstances use enumeration. \
# Example format: Government"""

def get_embeddings(text, api_key="AIzaSyCw1Pr_Uo2rLHu1uznHhKc6NhPxcwOpYkQ"):

    genai.configure(api_key=api_key)
    embedding = genai.embed_content(model="models/embedding-001",
                                    content=text,
                                    task_type="CLUSTERING")
    return embedding['embedding']

def summarize_text(prompt, api_key="AIzaSyCw1Pr_Uo2rLHu1uznHhKc6NhPxcwOpYkQ"):
    genai.configure(api_key=api_key)

    # Set up the model
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    response = model.generate_content(prompt)

    return response

def build_faiss_index(embeddings):
        embeddings = np.array(embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

def project(embeddings):
        mapper = UMAP(n_components=2, metric="cosine").fit(
            embeddings
        )
        return mapper.embedding_, mapper

def cluster(embeddings):
        clustering = DBSCAN(
            eps=0.08,
            min_samples=50,
            n_jobs=16,
        ).fit(embeddings)

        return clustering.labels_

def _postprocess_response(response):
    try:
        text = response.text

        # summary = ",".join(
        #             [txt for txt in text.strip().split(",") if len(txt) > 0]
        #         )
        summary = text.strip()
        
        return summary
    except Exception as e:
        return "NOT DETECT YET"


def summarize(texts, labels, label2docs,summary_n_examples=15):
        unique_labels = len(set(labels)) - 1  # exclude the "-1" label
        cluster_summaries = {-1: "None"}
        count = 0
        for label in range(unique_labels):
            ids = np.random.choice(label2docs[label], summary_n_examples)
            examples = "\n\n".join(
                [
                    f"Example {i+1}:\n{texts[_id][:5000]}"
                    for i, _id in enumerate(ids)
                ]
            )

            request = summary_template.format(examples)
            response = summarize_text(request)
            # if not response.text:
                # response = summarize_text(request)
            print(response)
            # if response.text:
            cluster_summaries[label] = _postprocess_response(response)
            
                 
        print(f"Number of clusters is {len(cluster_summaries)}")
        return cluster_summaries

def fit(texts, embeddings, summary_create=True):
        texts = texts
        embeddings = embeddings

        faiss_index = build_faiss_index(embeddings)
        faiss.write_index(faiss_index, f"./output/faiss.index")

        logging.info("projecting with umap...")
        projections, umap_mapper = project(embeddings)
        with open(f"./output/projections.npy", "wb") as f:
            np.save(f, projections)

        logging.info("dbscan clustering...")
        cluster_labels = cluster(projections)
        with open(f"./output/cluster_labels.npy", "wb") as f:
            np.save(f, cluster_labels)

        id2cluster = {
            index: label for index, label in enumerate(cluster_labels)
        }
        label2docs = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            label2docs[label].append(i)

        cluster_centers = {}
        for label in label2docs.keys():
            x = np.mean([projections[doc, 0] for doc in label2docs[label]])
            y = np.mean([projections[doc, 1] for doc in label2docs[label]])
            cluster_centers[label] = (x, y)

        if summary_create:
            logging.info("summarizing cluster centers...")
            cluster_summaries = summarize(texts, cluster_labels, label2docs)
        else:
            cluster_summaries = None
        
        #save results
        if cluster_summaries:
            with open(f"./output/cluster_summaries.json", "w") as f:
                json.dump(cluster_summaries,ensure_ascii=False, fp=f)

        return embeddings, cluster_labels, cluster_summaries


def load(folder="output"):
    

    faiss_index = faiss.read_index(f"./{folder}/faiss.index")

    with open(f"./{folder}/projections.npy", "rb") as f:
        projections = np.load(f)

    with open(f"./{folder}/cluster_labels.npy", "rb") as f:
        cluster_labels = np.load(f)


    if os.path.exists(f"./{folder}/cluster_summaries.json"):
        with open(f"./{folder}/cluster_summaries.json", "r") as f:
            cluster_summaries = json.load(f)
            keys = list(cluster_summaries.keys())
            for key in keys:
                cluster_summaries[int(key)] = cluster_summaries.pop(key)
    else:
        cluster_summaries = None

    # those objects can be inferred and don't need to be saved/loaded
    id2cluster = {
        index: label for index, label in enumerate(cluster_labels)
    }
    label2docs = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        label2docs[label].append(i)

    cluster_centers = {}
    for label in label2docs.keys():
        x = np.mean([projections[doc, 0] for doc in label2docs[label]])
        y = np.mean([projections[doc, 1] for doc in label2docs[label]])
        cluster_centers[label] = (x, y)
    
    return faiss_index, projections, cluster_labels, cluster_summaries, id2cluster, label2docs, cluster_centers

def create_summary(texts, cluster_labels, label2docs, save=False):
    cluster_summaries = summarize(texts, cluster_labels, label2docs)
    if save:
        with open(f"./output/cluster_summaries.json", "w") as f:
            json.dump(cluster_summaries,ensure_ascii=False, fp=f)
    return cluster_summaries

def infer(text, faiss_index, cluster_labels, top_k=1 ):
    ###Embeddings is list of embeddings###
    embeddings = get_embeddings(text)
    embeddings = np.array([embeddings])
    print(embeddings.shape  )
    dist, neighbours = faiss_index.search(embeddings, top_k)
    inferred_labels = []
    for i in tqdm(range(embeddings.shape[0])):
        labels = [cluster_labels[doc] for doc in neighbours[i]]
        inferred_labels.append(Counter(labels).most_common(1)[0][0])

    return inferred_labels


def show(texts,projections,cluster_labels,cluster_summaries, cluster_centers, interactive=False):
        df = pd.DataFrame(
            data={
                "X": projections[:, 0],
                "Y": projections[:, 1],
                "labels": cluster_labels,
                "content_display": [
                    textwrap.fill(txt[:1024], 64) for txt in texts
                ],
            }
        )

        if interactive:
            _show_plotly(df, cluster_summaries, cluster_centers)
        else:
            _show_mpl(df, cluster_summaries, cluster_centers)

def _show_mpl( df, cluster_summaries, cluster_centers):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    df["color"] = df["labels"].apply(lambda x: "C0" if x==-1 else f"C{(x%9)+1}")

    df.plot(
        kind="scatter",
        x="X",
        y="Y",
        # c="labels",
        s=0.75,
        alpha=0.8,
        linewidth=0,
        color=df["color"],
        ax=ax,
        colorbar=False,
    )

    for label in cluster_summaries.keys():
        if label == -1:
            continue
        summary = cluster_summaries[label]
        position = cluster_centers[label]
        t= ax.text(
            position[0],
            position[1],
            summary,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=4,
        )
        t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))
    ax.set_axis_off()
    fig.savefig('./output/figure.png')

def _show_plotly( df, cluster_summaries, cluster_centers):
    fig = px.scatter(
        df,
        x="X",
        y="Y",
        color="labels",
        hover_data={"content_display": True, "X": False, "Y": False},
        width=1600,
        height=800,
        color_continuous_scale="HSV",
    )

    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_traces(
        marker=dict(size=1, opacity=0.8),  # color="white"
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        template="plotly_dark",
    )

    # show cluster summaries
    for label in cluster_summaries.keys():
        if label == -1:
            continue
        summary = cluster_summaries[label]
        position = cluster_centers[label]

        fig.add_annotation(
            x=position[0],
            y=position[1],
            text=summary,
            showarrow=False,
            yshift=0,
        )

    fig.show()