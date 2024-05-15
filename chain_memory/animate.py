from typing import List
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import textwrap


def create_scatter_object(coordinates, system_colors, hovertext, mode, i=None):
    """Helper function to create a Scatter3D object"""
    x, y, z, _, = zip(*coordinates[:i]) if i is not None else zip(*coordinates)

    if mode == "markers":
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode=mode,
            marker=dict(
                size=5,
                color=system_colors[:i] if i is not None else system_colors,
                colorscale="rainbow",
                colorbar=dict(title="Labels", x=-0.1, xanchor="left"),
                line=dict(color="powderblue", width=2),
                opacity=0.9,
                symbol="circle",
                sizemode="diameter",
            ),
            hoverinfo="text",
            hovertext=hovertext[:i] if i is not None else hovertext,
            name="Coordinates",
        )
    else:  # mode == "lines"
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode=mode,
            line=dict(color="white", width=1),
            hoverinfo="none",
        )


def animate_conversation_tree(coordinates: List[object], name: str = "Chain Tree"):
    """Create and animate a 3D scatter plot of conversation coordinates."""
    hovertext = [
        "<br><br>".join(
            textwrap.wrap(
                f"Depth: {coord[0]}\n\nSibling: {coord[1]}\n\nSibling Count: {coord[2]}\n\nAuthor: {coord[3]}",
                width=100,  # you can adjust the width as needed
            )
        )
        for coord in coordinates
    ]

    rainbow_scale = px.colors.sequential.Rainbow
    system_colors = [
        rainbow_scale[int(i)]
        for i in np.linspace(0, len(rainbow_scale) - 1, len(coordinates))
    ]

    frames = [
        go.Frame(
            data=[
                create_scatter_object(
                    coordinates, system_colors, hovertext, "markers", i
                ),
                create_scatter_object(
                    coordinates, system_colors, hovertext, "lines", i
                ),
            ]
        )
        for i in range(1, len(coordinates) + 1)
    ]

    fig = go.Figure(
        data=[
            create_scatter_object(coordinates, system_colors, hovertext, "markers"),
            create_scatter_object(coordinates, system_colors, hovertext, "lines"),
        ],
        layout=go.Layout(
            title=name,
            scene=dict(
                xaxis=dict(showbackground=False, gridcolor="Black"),
                yaxis=dict(showbackground=False, gridcolor="Black"),
                zaxis=dict(showbackground=False, gridcolor="Black"),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(x=-1.5, y=-1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                ),
                dragmode="turntable",
            ),
            font_family="Arial",
            font_color="White",
            title_font_family="Arial",
            title_font_color="White",
            legend_title_font_color="White",
            paper_bgcolor="Black",
            plot_bgcolor="Black",
            hoverlabel=dict(bgcolor="Black", font_color="White"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(frame=dict(duration=500, redraw=True)),
                            ],
                        )
                    ],
                )
            ],
        ),
        frames=frames,
    )
    fig.show()



if __name__ == "__main__":
    # Load the conversation coordinates
    coordinates = np.load("chain_memory/data/coordinates_tree.npy", allow_pickle=True)

    animate_conversation_tree(coordinates)