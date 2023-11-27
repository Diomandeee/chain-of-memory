from typing import Union
from dash.dependencies import Input, Output
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import dash
import io
import base64


def wrap_line_to_max_length(line: str, max_length: int) -> list[str]:
    """
    Wraps a single line of text to the specified maximum length.

    Parameters:
        line: str
            The line of text to be wrapped.
        max_length: int
            The maximum length for each wrapped line.

    Returns:
        list[str]
            A list of wrapped lines.
    """
    wrapped_line = []
    while len(line) > max_length:
        cut_at = max_length

        # Find the last space within the max length
        space_index = line.rfind(" ", 0, max_length)

        if space_index != -1:
            cut_at = space_index

        wrapped_line.append(line[:cut_at].rstrip())  # Remove trailing spaces
        line = line[cut_at:].lstrip()  # Remove leading spaces
    wrapped_line.append(line)
    return wrapped_line


def wrap_text(text: str, max_length: int = 100, max_lines: int = 5) -> str:
    """
    Wraps and truncates the given text to the specified maximum length and number of lines, 
    and returns it separated by HTML break tags for double-spacing.

    Parameters:
        text: str
            The text to be wrapped and truncated.
        max_length: int
            The maximum length for each wrapped line.
        max_lines: int
            The maximum number of lines to include.

    Returns:
        str
            The wrapped and truncated text separated by double HTML break tags.
    """
    lines = text.split("\n")
    wrapped_lines = [wrap_line_to_max_length(line, max_length) for line in lines]
    flattened_lines = [line for sublist in wrapped_lines for line in sublist]

    # Truncate the text if it exceeds the maximum number of lines
    truncated_lines = flattened_lines[:max_lines]

    # Optionally, you can add an ellipsis to indicate that the text has been truncated
    if len(flattened_lines) > max_lines:
        truncated_lines[-1] += '...'

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
        "n_parts",
        "author",
        "create_time",
        "cluster_label",
        "n_neighbors",
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


def agg_funcs():
    aggs = {
        "count": len,
        "sum": np.sum,
        "avg": np.mean,
        "median": np.median,
        "rms": lambda x: np.sqrt(np.mean(np.square(x))),
        "stddev": np.std,
        "min": np.min,
        "max": np.max,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1],
    }
    return aggs


def aggregate_data(
    df: pd.DataFrame, label_column: str, columns_to_agg: list, aggregate_func: callable
) -> pd.DataFrame:
    if aggregate_func is not None:
        if aggregate_func not in agg_funcs().keys():
            raise ValueError(
                f"Invalid aggregate function. Choose from {list(agg_funcs().keys())}"
            )
        grouped = df.groupby(label_column)
        aggregated_data = (
            grouped[columns_to_agg].agg(agg_funcs()[aggregate_func]).reset_index()
        )
        return pd.merge(
            df, aggregated_data, on=label_column, how="left", suffixes=("", "_agg")
        )
    else:
        return df

def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if "csv" in content_type:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    else:
        return None


def create_dash_app(data, text_column="text"):
    # Check if the input is a path or a DataFrame
    if isinstance(data, str):
        try:
            # Attempt to read the CSV file, skipping the problematic row
            df = pd.read_csv(data, error_bad_lines=False)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
    elif isinstance(data, pd.DataFrame):
        # If it's a DataFrame, use it directly
        df = data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame")

    original_df = prepare_dataframe(df, text_column)
    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={
            "width": "100%",
            "height": "100vh",
            "display": "flex",
            "flex-direction": "column",
            "background-color": "#f4f4f4",
        },
        children=[
            dcc.Graph(
                id="3d_scatter",
                style={"height": "90vh"},
            ),
            html.Div( 
                style={
                    "display": "flex",
                    "flex-direction": "row",
                    "align-items": "center",
                    "margin-bottom": "20px",
                    "justify-content": "space-between",
                },
                children=[
                    html.Div(  
                        children=[
                            html.Div(
                                "X:",
                                style={
                                    "flex-grow": "1",
                                    "padding": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="depth_x_condition_input",
                                type="text",
                                placeholder="e.g., depth_x == 1",
                                value="depth_x ",
                                style={"width": "170px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "padding": "10px",
                        },
                    ),
                    html.Div(  
                        children=[
                            html.Div(
                                "Y:",
                                style={
                                    "flex-grow": "1",
                                    "padding": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="sibling_y_condition_input",
                                type="text",
                                placeholder="e.g., sibling_y < 5",
                                value="sibling_y ",
                                style={"width": "170px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "padding": "10px",
                        },
                    ),
                    html.Div( 
                        children=[
                            html.Div(
                                "Z:",
                                style={
                                    "flex-grow": "1",
                                    "padding": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="sibling_count_z_condition_input",
                                type="text",
                                placeholder="e.g., sibling_count_z >= 3",
                                value="sibling_count_z ",
                                style={"width": "170px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "padding": "10px",
                        },
                    ),
                    html.Div( 
                        children=[
                            html.Div(
                                "N:",
                                style={
                                    "flex-grow": "1",
                                    "padding": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="n_part_input",
                                type="text",
                                placeholder="e.g., n_part >= 3",
                                value="n_parts ",
                                style={"width": "170px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "padding": "10px",
                        },
                    ),
                    html.Div( 
                        children=[
                            html.Div(
                                "T:",
                                style={
                                    "flex-grow": "1",
                                    "padding": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="time_t_input",
                                type="text",
                                placeholder="e.g., time_t >= 1",
                                value="time_t ",
                                style={"width": "170px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "padding": "10px",
                        },
                    ),
                    html.Div(  
                        children=[
                            html.Div(
                                "Label:",
                                style={
                                    "flex-grow": "1",
                                    "padding": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            dcc.Input(
                                id="label",
                                type="text",
                                placeholder="e.g., cluster_label == 1",
                                value="cluster_label ",
                                style={"width": "170px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "padding": "10px",
                        },
                    ),
                ],
            ),
            html.Details(
                [
                    html.Summary(
                        "Load Data",
                        style={"font-weight": "bold"},
                    ),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drag and Drop or ", html.A("Select Files")]
                        ),
                        style={
                            "width": "100%",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                        },
                        multiple=False,
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("3d_scatter", "figure"),
        [
            Input("upload-data", "contents"),
            Input("depth_x_condition_input", "value"),
            Input("sibling_y_condition_input", "value"),
            Input("sibling_count_z_condition_input", "value"),
            Input("n_part_input", "value"),
            Input("time_t_input", "value"),
            Input("label", "value"),
        ],
    )
    def update_figure(
        contents,
        depth_x_condition,
        sibling_y_condition,
        sibling_count_z_condition,
        n_part,
        time_t,
        label,
    ):
        # Use original DataFrame if no file is uploaded
        if contents:
            df = parse_contents(contents)
            if df is None:
                raise dash.exceptions.PreventUpdate
            df = prepare_dataframe(df, text_column)
        else:
            df = original_df.copy()  # Use a copy of the original DataFrame

        # Apply conditions if provided
        for condition, column in zip(
            [
                depth_x_condition,
                sibling_y_condition,
                sibling_count_z_condition,
                n_part,
                time_t,
                label,
            ],
            [
                "depth_x",
                "sibling_y",
                "sibling_count_z",
                "n_parts",
                "time_t",
                "cluster_label",
            ],
        ):
            if condition and condition.strip() != column:
                try:
                    df = df.query(condition)
                except Exception as e:
                    pass  # Ignore errors in condition application

        return plot_3d_scatter(df)

    return app


path = "data/example.csv"


df = pd.read_csv(path, skiprows=[1186])
app = create_dash_app(df)

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8051)

# pip requirements
