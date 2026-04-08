import plotly.graph_objects as go

def plot_topics_3d(c_vecs_all, m_vecs_all, c_keywords_all, m_keywords_all, c_weights, m_weights):
    all_vecs = np.vstack(c_vecs_all + m_vecs_all)
    reducer = umap.UMAP(n_components=3, n_neighbors=5, random_state=42, metric="cosine")
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

    # Find closest modern topic for each classical topic
    sim_matrix = cosine_similarity(np.vstack(c_vecs_all), np.vstack(m_vecs_all))
    closest_modern = sim_matrix.argmax(axis=1)

    # Build connection lines as a single trace with None separators
    line_x, line_y, line_z = [], [], []
    for i, j in enumerate(closest_modern):
        line_x += [c_coords[i, 0], m_coords[j, 0], None]
        line_y += [c_coords[i, 1], m_coords[j, 1], None]
        line_z += [c_coords[i, 2], m_coords[j, 2], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color='#cccccc', width=1),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=c_coords[:, 0], y=c_coords[:, 1], z=c_coords[:, 2],
        mode='markers+text',
        name='Classical',
        marker=dict(size=c_sizes, color='#7F77DD', opacity=0.85),
        text=[' '.join(c_keywords_all[i][:2]) for i in range(n_c)],
        textposition='top center',
        textfont=dict(size=10, color='#534AB7'),
        hovertext=c_hover,
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter3d(
        x=m_coords[:, 0], y=m_coords[:, 1], z=m_coords[:, 2],
        mode='markers+text',
        name='Modern',
        marker=dict(size=m_sizes, color='#1D9E75', opacity=0.85),
        text=[' '.join(m_keywords_all[j][:2]) for j in range(len(m_vecs_all))],
        textposition='top center',
        textfont=dict(size=10, color='#0F6E56'),
        hovertext=m_hover,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='Classical vs Modern Topic Space (UMAP 3D)',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='white',
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1000,
        height=800,
    )

    fig.write_html('topics_3d.html')
    print("Saved: topics_3d.html")
    fig.show()