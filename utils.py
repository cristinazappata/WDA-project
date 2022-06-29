import plotly.graph_objects as go
import networkx as nx
from typing import Any
import json
from typing import Union
import matplotlib.pyplot as plt

import numpy as np
#import plotly.io as pio
#pio.renderers.default = "browser"
#matplotlib.rcParams['figure.dpi'] = 300


def network_plot(G, pos=None, description=None):
    if pos is None:
        pos = nx.fruchterman_reingold_layout(G)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]  # type: ignore
        x1, y1 = pos[edge[1]]  # type: ignore
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.75, color='rgba(70, 70, 70, 0.25)'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []

    node_text = []

    for node in G.nodes():
        x, y = pos[node] # type: ignore
        node_x.append(x)
        node_y.append(y)
        if description:
            text = description[node]
        else:
            text = node
        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='viridis',
            reversescale=False,
            color=[],
            size=8,
            line=dict(
                width=1
            )))

    node_adjacencies = []

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies # type: ignore
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False)))

    return fig


def load_dataset(dataset_path: str) -> Any:
    with open(dataset_path) as f:
        dataset = json.load(f)
    return dataset


def store_dataset(dataset, out_file_path: str):
    with open(out_file_path, 'w') as f:
        json.dump(dataset, f)


def top_k_sim_docs(corpus_mtx, query_idx, k=5, with_sims=False):
        query = corpus_mtx[query_idx]
        sim_score = np.dot(corpus_mtx, query) / (np.linalg.norm(corpus_mtx, axis=1) * np.linalg.norm(query))
        sim_score_sorted_idx = np.argsort(sim_score)[::-1]
        sim_score_sorted_idx = sim_score_sorted_idx[:k]
        return (sim_score_sorted_idx, sim_score[sim_score_sorted_idx]) if with_sims else sim_score_sorted_idx

def print_top_sim_docs(corpus_mtx, query_idx, captions, k=5):
    idxs, sims = top_k_sim_docs(corpus_mtx, query_idx, k, with_sims=True)
    for i, s in zip(idxs, sims):
        print(f"!{i}! ----------- {s}")
        label = captions[i][:300]
        print(f"{label}...")
        print("")

def plot_eigenvalues(values):
    plt.plot(values[::-1])


def threshold_sim_docs(corpus_mtx, query_idx, threshold):
    """
    Funzione che ricevuti degli elementi rappresentati da righe di una matrice, un indice ed una soglia
    restituisce gli elementi che hanno un similarità coseno maggiore o uguale a threshold rispetto 
    all'elemento di indice query_idx.
    """
    query = corpus_mtx[query_idx]
    sim_score = np.dot(corpus_mtx, query) / (np.linalg.norm(corpus_mtx, axis=1) * np.linalg.norm(query))
    idxs = np.argwhere(sim_score > threshold)
    return np.vstack([np.squeeze(idxs), np.squeeze(sim_score[idxs])]).T


def docs_to_graph(corpus_mtx, threshold):
    """
    Funzione che ricevuti degli elementi rappresentati da righe di una matrice e una soglia
    restituisce un grafo in cui i nodi corrispondono a gli elementi nelle righe della matrice e
    gli archi rappresentano una similarità coseno tra i vettori in questione maggiore o uguale a threshold.
    """
    G = nx.Graph()
    edges = []
    for idx in range(corpus_mtx.shape[0]):
        for edge in threshold_sim_docs(corpus_mtx, idx, threshold):
            edges.append((idx, int(edge[0]), edge[1]))
    G.add_weighted_edges_from(edges)
    return G