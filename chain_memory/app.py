from dash.dependencies import Input, Output
from dash import html, dcc
from plot import plot_3d_scatter, prepare_dataframe
import pandas as pd
import numpy as np
import dash
import io
import base64
import argparse


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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Dash app")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the Dash app on"
    )
    parser.add_argument(
        "--port", type=int, default=8051, help="Port to run the Dash app on"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Read the DataFrame from the specified path
    path = "chain_memory/data/example.csv"
    df = pd.read_csv(path)

    # Create and run the Dash app
    app = create_dash_app(df)
    app.run_server(host=args.host, port=args.port)
