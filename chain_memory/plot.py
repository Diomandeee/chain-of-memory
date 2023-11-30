from typing import Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from textwrap import wrap


def wrap_line_to_max_length(line: str, max_length: int) -> list:
    """
    Wraps a single line of text to the specified maximum length.

    Parameters:
        line: str
            The line to be wrapped.
        max_length: int
            The maximum length for each wrapped line.

    Returns:
        list
            A list of wrapped lines.
    """
    return wrap(line, max_length)


def wrap_text(
    text: str,
    max_length: int = 100,
    max_lines: int = float("inf"),
    ellipsis: bool = False,
) -> str:
    """
    Wraps and truncates the given text to the specified maximum length and number of lines,
    and returns it separated by HTML break tags for double-spacing.

    Parameters:
        text: str
            The text to be wrapped and truncated.
        max_length: int
            The maximum length for each wrapped line.
        max_lines: int
            The maximum number of lines to include. Defaults to showing all lines.
        ellipsis: bool
            Flag to add ellipsis if text is truncated.

    Returns:
        str
            The wrapped and truncated text separated by double HTML break tags.
    """
    lines = text.split("\n")
    wrapped_lines = [wrap_line_to_max_length(line, max_length) for line in lines]
    flattened_lines = [line for sublist in wrapped_lines for line in sublist]

    # Handle infinite max_lines
    if max_lines == float("inf"):
        truncated_lines = flattened_lines
    else:
        truncated_lines = flattened_lines[:max_lines]
        # Add ellipsis if text is truncated
        if ellipsis and len(flattened_lines) > max_lines:
            truncated_lines[-1] += "..."

    return "<br><br>".join(truncated_lines)


def prepare_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    def custom_round(value):
        if value >= 0.9999959208612414:
            return 1
        else:
            return round(value, 4)

    # Drop the 'embeddings' column if it exists
    df.drop(columns=["embeddings"], errors="ignore", inplace=True)

    # Ensure the text column is of string type
    df[text_column] = df[text_column].astype(str)

    # Apply any text formatting function
    df["formatted_content"] = df[text_column].apply(wrap_text)

    if "create_time" in df.columns:
        df["create_time"] = pd.to_datetime(df["create_time"], unit="s")

    # Convert string representations of lists to actual lists
    if "coordinate" in df.columns:
        df["coordinate"] = df["coordinate"].apply(eval)

        # Expand the 'coordinate' column into separate columns
        coord_columns = ["depth_x", "sibling_y", "sibling_count_z", "time_t", "n_parts"]
        coords_df = pd.DataFrame(df["coordinate"].tolist(), columns=coord_columns)

        # Apply custom rounding to 'time_t'
        coords_df["time_t"] = coords_df["time_t"].apply(custom_round)

        df = pd.concat([df.drop(columns=["coordinate"]), coords_df], axis=1)

    if "umap_embeddings" in df.columns:
        df["umap_embeddings"] = df["umap_embeddings"].apply(eval)

        # Expand the 'umap_embeddings' column into separate columns
        umap_columns = ["x", "y", "z"]
        umap_df = pd.DataFrame(df["umap_embeddings"].tolist(), columns=umap_columns)
        df = pd.concat([df.drop(columns=["umap_embeddings"]), umap_df], axis=1)

    if "n_neighbors" not in df.columns:
        df["n_neighbors"] = 0

    return df


def add_coordinate_hover_info(
    df: pd.DataFrame, custom_hover_info: dict = None
) -> pd.DataFrame:
    if custom_hover_info is None:
        custom_hover_info = {}

    # Function to build hover text based on available columns
    def build_hover_text(row, column_names):
        hover_texts = []

        for col in column_names:
            if col in row:
                hover_texts.append(f"{col}: {row[col]}")

        return "<br>".join(hover_texts)

    def to_string(value):
        return str(value)

    base_cols = [
        "time_t",
        "author",
        "create_time",
        "formatted_content",
    ]

    # Default hover info for UMAP
    if "formatted_content_umap" not in custom_hover_info:
        umap_cols = ["x", "y", "z"] + base_cols
        df["formatted_content_umap"] = df.apply(
            lambda row: build_hover_text(row, umap_cols), axis=1
        )

    # Default hover info for Coordinates
    if "formatted_content_coord" not in custom_hover_info and "depth_x" in df.columns:
        coord_cols = ["depth_x", "sibling_y", "sibling_count_z"] + base_cols
        df["formatted_content_coord"] = df.apply(
            lambda row: build_hover_text(row, coord_cols), axis=1
        )

    # Applying custom hover info
    for column, hover_format in custom_hover_info.items():
        df[column] = df.apply(lambda row: hover_format.format(**row), axis=1)

    df["n_neighbors"] = df["n_neighbors"].apply(to_string)

    return df


def generate_traces(df: pd.DataFrame, system_colors: list, colorscale: str) -> list:
    """
    Generate traces for plotting.

    Args:
        df (pd.DataFrame): The input DataFrame.
        system_colors (list): The list of system colors.
        colorscale (str): The colorscale for the plot.

    Returns:
        list: The list of traces for plotting.
    """
    scatter_trace_umap = generate_scatter_trace(
        df, "x", "y", "z", "formatted_content_umap", system_colors, colorscale
    )
    line_trace_umap = generate_line_trace(df, "x", "y", "z")

    if "depth_x" in df.columns:
        scatter_trace_coord = generate_scatter_trace(
            df,
            "depth_x",
            "sibling_y",
            "sibling_count_z",
            "formatted_content_coord",
            system_colors,
            colorscale,
        )

        line_trace_coord = generate_line_trace(
            df, "depth_x", "sibling_y", "sibling_count_z"
        )
        return [
            scatter_trace_umap,
            line_trace_umap,
            scatter_trace_coord,
            line_trace_coord,
        ]
    else:
        return [scatter_trace_umap, line_trace_umap]


def generate_colors(df: pd.DataFrame, label_column: str) -> list:
    """
    Generate a list of colors for the markers in the plot.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        label_column (str): The column name for the cluster labels.

    Returns:
        list: The list of colors for the markers.
    """

    rainbow_scale = px.colors.sequential.Rainbow
    return [
        rainbow_scale[int(i)]
        for i in np.linspace(0, len(rainbow_scale) - 1, len(df[label_column]))
    ]


def generate_scatter_trace(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    hover_text: str,
    colors: list,
    colorscale: str,
) -> go.Scatter3d:
    """
    Generate a scatter trace for a 3D scatter plot.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis values.
        y_col (str): The column name for the y-axis values.
        z_col (str): The column name for the z-axis values.
        hover_text (str): The column name for the hover text values.
        colors (list): The list of colors for the markers.
        colorscale (str): The colorscale for the markers.

    Returns:
        go.Scatter3d: The scatter trace for the 3D scatter plot.
    """
    return go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode="markers",
        marker=dict(
            size=4,
            color=colors,
            colorscale=colorscale,
            line=dict(color="powderblue", width=2),
            opacity=0.7,
            symbol="circle",
            sizemode="diameter",
        ),
        hoverinfo="text",
        hovertext=df[hover_text],
    )


def generate_line_trace(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str
) -> go.Scatter3d:
    """
    Generate a 3D line trace for a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis values.
        y_col (str): The column name for the y-axis values.
        z_col (str): The column name for the z-axis values.

    Returns:
        go.Scatter3d: The 3D line trace.
    """
    return go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode="lines",
        line=dict(
            color="white",
            colorscale="Rainbow",
            width=1,
            cmin=0,
            cmax=1,
        ),
        hoverinfo="none",
    )


def generate_layout(
    dragmode: str = "turntable",
    result3d: pd.DataFrame = None,
    title: str = "Chain of Memories",
    title_font_size: int = 20,
) -> go.Layout:
    """
    Generate the layout for the Chain of Memories plot.

    Parameters:
        dragmode (str): The drag mode for the plot. Default is "turntable".
        result3d (pd.DataFrame): The 3D result data. Default is None.
        title (str): The title of the plot. Default is "Chain of Memories".
        title_font_size (int): The font size of the title. Default is 20.

    Returns:
        go.Layout: The generated layout for the plot.
    """
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=list(
                [
                    dict(
                        args=[{"visible": [True, True, True, True]}],
                        label="Show All",
                        method="update",
                    ),
                    dict(
                        args=[{"visible": [True, True, False, False]}],
                        label="Show UMAP",
                        method="update",
                    ),
                    dict(
                        args=[{"visible": [False, False, True, True]}],
                        label="Show Coordinates",
                        method="update",
                    ),
                ]
            ),
            direction="down",
            pad={"r": 10, "t": 10},
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top",
        ),
    ]

    layout = dict(
        title=dict(
            text=title,
            font=dict(size=title_font_size),
            x=0.5,
            xanchor="center",
        ),
        showlegend=True,
        updatemenus=updatemenus,
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor="Black"),
            yaxis=dict(showbackground=False, gridcolor="Black"),
            zaxis=dict(showbackground=False, gridcolor="Black"),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=-1.5, y=-1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
            ),
            dragmode=dragmode,
        ),
        font_family="Arial",
        font_color="White",
        title_font_family="Arial",
        title_font_color="White",
        legend_title_font_color="White",
        paper_bgcolor="Black",
        plot_bgcolor="Black",
        hoverlabel=dict(bgcolor="Black", font_color="White"),
        coloraxis_colorbar=(
            go.layout.Colorbar(
                title="Color Scale",
                titleside="right",
                tickmode="array",
                tickvals=[min(result3d["labels"]), max(result3d["labels"])],
                ticktext=["Low", "High"],
            )
            if "color" in result3d.columns
            else None
        ),
    )

    return layout


def plot_3d_scatter(
    file_path_or_dataframe: Union[str, pd.DataFrame],
    text_column: str = "text",
    label_column: str = "cluster_label",
    colorscale: str = "Rainbow",
    dragmode: str = "turntable",
    show: bool = False,
):
    """
    Plots a 3D scatter plot using the provided file path or dataframe.

    Parameters:
    - file_path_or_dataframe (Union[str, pd.DataFrame]): The file path or dataframe containing the data.
    - text_column (str): The column name in the dataframe that contains the text data.
    - label_column (str): The column name in the dataframe that contains the cluster labels.
    - colorscale (str): The colorscale to use for the scatter plot.
    - dragmode (str): The drag mode for the plot.
    - show (bool): Whether to show the plot or return the figure object.

    Returns:
    - fig (go.Figure): The figure object representing the 3D scatter plot.
    """
    df = file_path_or_dataframe
    df = prepare_dataframe(df, text_column)
    system_colors = generate_colors(df, label_column)
    df = add_coordinate_hover_info(df)
    traces = generate_traces(df, system_colors, colorscale)
    layout = generate_layout(dragmode, df)
    fig = go.Figure(data=traces, layout=layout)

    if show:
        fig.show()
    else:
        return fig
