import plotly.graph_objects as go
import numpy as np
import umap
from sklearn.metrics.pairwise import cosine_similarity
import flask

def build_figure(c_vecs_all, m_vecs_all, c_keywords_all, m_keywords_all, c_weights, m_weights):
    all_vecs = np.vstack(c_vecs_all + m_vecs_all)

    reducer = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.8,
                        spread=3.0, random_state=42, metric="cosine")
    coords = reducer.fit_transform(all_vecs)

    n_c = len(c_vecs_all)
    c_coords = coords[:n_c]
    m_coords = coords[n_c:]

    def scale_sizes(weights, s_min=8, s_max=30):
        w = np.array(weights, dtype=float)
        w_min, w_max = w.min(), w.max()
        if w_max - w_min < 1e-9:
            return np.full(len(w), (s_min + s_max) / 2)
        return s_min + (s_max - s_min) * (w - w_min) / (w_max - w_min)

    c_sizes = scale_sizes(c_weights)
    m_sizes = scale_sizes(m_weights)

    c_hover = [f"Classical {i+1}<br>{'<br>'.join(c_keywords_all[i][:5])}<br>weight={c_weights[i]:.4f}"
               for i in range(n_c)]
    m_hover = [f"Modern {j+1}<br>{'<br>'.join(m_keywords_all[j][:5])}<br>weight={m_weights[j]:.4f}"
               for j in range(len(m_vecs_all))]

    sim_matrix = cosine_similarity(np.vstack(c_vecs_all), np.vstack(m_vecs_all))
    closest_modern = sim_matrix.argmax(axis=1)

    line_x, line_y = [], []
    for i, j in enumerate(closest_modern):
        line_x += [c_coords[i, 0], m_coords[j, 0], None]
        line_y += [c_coords[i, 1], m_coords[j, 1], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode='lines',
        line=dict(color='rgba(53, 6, 62, 0.4)', width=0.5),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=c_coords[:, 0], y=c_coords[:, 1],
        mode='markers+text',
        name='Classical',
        marker=dict(size=c_sizes, color='#7F77DD', opacity=0.85),
        text=[' '.join(c_keywords_all[i][:2]) for i in range(n_c)],
        textposition='top center',
        textfont=dict(size=12, color='#35063e'),
        hovertext=c_hover,
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=m_coords[:, 0], y=m_coords[:, 1],
        mode='markers+text',
        name='Modern',
        marker=dict(size=m_sizes, color='#1D9E75', opacity=0.85),
        text=[' '.join(m_keywords_all[j][:2]) for j in range(len(m_vecs_all))],
        textposition='top center',
        textfont=dict(size=12, color='#35063e'),
        hovertext=m_hover,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Classical vs Modern Topic Space (UMAP 2D)',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=60, b=40),
        width=1000,
        height=800,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    )
    return fig


def plot_topics_2d(c_vecs_all, m_vecs_all, c_keywords_all, m_keywords_all, c_weights, m_weights):
    fig = build_figure(c_vecs_all, m_vecs_all, c_keywords_all, m_keywords_all, c_weights, m_weights)

    app = flask.Flask(__name__)

    @app.route('/')
    def index():
        return fig.to_html(full_html=True, include_plotlyjs='cdn')

    print("Opening at http://127.0.0.1:5050 — press Ctrl+C to stop")
    app.run(port=5050, debug=False, use_reloader=False)