from typing import List, Tuple, Optional
from llama_index.core import SimpleDirectoryReader
from plot import plot_3d_scatter
import pandas as pd
import argparse
import re


def split_text(text: str) -> List[str]:
    """
    Splits the text into lines based on a specific pattern.

    Args:
    text (str): The text to be split.

    Returns:
    List[str]: A list of split lines.
    """
    lines = re.split(r"(\d+\.\d+\.\d+\s)", text)

    lines = [line.strip() for line in lines if line is not None and line.strip()]

    return lines


def clean_text(text: str) -> str:
    """
    Cleans the text by keeping only lines that start with a specific pattern.

    Args:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    cleaned_text = "\n".join(
        line for line in text.split("\n") if re.match(r"^\d+(\.\d+)*\s", line)
    )
    return cleaned_text


def split_into_sublists(text: str) -> List[List[str]]:
    """
    Splits the text into sublists based on a specific pattern.

    Args:
    text (str): The text to be split.

    Returns:
    List[List[str]]: A list of sublists.
    """
    grouped_text = []
    current_group = []

    for line in text.split("\n"):
        if re.match(r"^\d+(\.\d+)*\s", line):
            # A line with a number, start a new group
            if current_group:
                grouped_text.append(current_group)
            current_group = [line]
        else:
            # A continuation of the previous line, add it to the current group
            current_group.append(line)

    # Append the last group
    if current_group:
        grouped_text.append(current_group)

    return grouped_text


def create_hierarchy(grouped_text: List[List[str]]) -> dict:
    """
    Creates a hierarchical structure from the grouped text.

    Args:
    grouped_text (List[List[str]]): The grouped text.

    Returns:
    dict: The hierarchical structure.
    """
    hierarchical_structure = {}
    current_dict = hierarchical_structure

    for group in grouped_text:
        current_dict = hierarchical_structure
        for line in group:
            parts = re.match(r"^(\d+(\.\d+)*)\s(.*)$", line)
            if parts:
                number, _, content = parts.groups()
                number_parts = number.split(".")
                for part in number_parts:
                    current_dict = current_dict.setdefault(part, {})
                current_dict["content"] = content
                page_match = re.search(r"(\d+)$", content)
                if page_match:
                    current_dict["page"] = page_match.group()

    return hierarchical_structure


def create_cleaned_dataframe(hierarchical_structure: dict) -> pd.DataFrame:
    """
    Creates a cleaned dataframe from the hierarchical structure.

    Args:
    hierarchical_structure (dict): The hierarchical structure.

    Returns:
    pd.DataFrame: The cleaned dataframe.
    """

    def flatten_hierarchy(node, prefix=""):
        flat_dict = {}
        for key, value in node.items():
            if key == "content" and "page" in node:
                page = node["page"]
                flat_dict[prefix + "content"] = f"{value} (Page {page})"
            elif key == "content":
                flat_dict[prefix + "content"] = value
            elif key == "page":
                pass
            else:
                flat_dict.update(flatten_hierarchy(value, prefix + key + "."))
        return flat_dict

    flat_dict = flatten_hierarchy(hierarchical_structure)
    df = pd.DataFrame([flat_dict]).transpose().reset_index()
    df.columns = ["Hierarchy", "Content"]

    df["Hierarchy"] = df["Hierarchy"].str.replace(r"\.content", "", regex=True)
    df["Page"] = df["Content"].str.extract(r"Page (\d+)", expand=False)
    df["Content"] = df["Content"].str.replace(r"\d+\s*\(Page \d+\)", "", regex=True)

    return df


def get_pdf(pdf_path: str) -> SimpleDirectoryReader:
    """
    Loads the PDF file from the given path.

    Args:
    pdf_path (str): The path to the PDF file.

    Returns:
    SimpleDirectoryReader: The loaded PDF file.
    """
    pdf = SimpleDirectoryReader(
        input_files=[pdf_path],
    ).load_data()

    return pdf


def split_pdf(pdf: SimpleDirectoryReader, start_page: int, end_page: int) -> List:
    """
    Splits the PDF into selected pages.

    Args:
    pdf (SimpleDirectoryReader): The PDF file.
    start_page (int): The starting page.
    end_page (int): The ending page.

    Returns:
    List: The selected pages.
    """

    selected_pages = pdf[start_page:end_page]

    return selected_pages


def append_text(selected_pages: List) -> Tuple[pd.DataFrame, List]:
    """
    Processes the selected pages and appends the text.

    Args:
    selected_pages (List): The selected pages.

    Returns:
    Tuple[pd.DataFrame, List]: A tuple containing the dataframe and sub-groups.
    """
    text_lines = []
    for page in selected_pages:
        text = page.text
        text_lines.extend(split_text(text))

    new_text = "\n".join(text_lines)
    cleaned_text = clean_text(new_text)
    grouped_text = split_into_sublists(cleaned_text)
    hierarchical_structure = create_hierarchy(grouped_text)
    df = create_cleaned_dataframe(hierarchical_structure)
    return df


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the dataframe with additional columns and adjustments.

    Args:
    df (pd.DataFrame): The dataframe to format.

    Returns:
    pd.DataFrame: The formatted dataframe.
    """
    # loop through each value in the 'column_name' column
    for i in range(len(df["Hierarchy"])):
        # if the value is does not contain'.', add a '0' after the '.'
        if "." not in df["Hierarchy"][i]:
            df["Hierarchy"][i] = df["Hierarchy"][i] + ".0"

    df["x"] = df["Hierarchy"].str.split(".").str[0].astype(int)

    df["y"] = df["Hierarchy"].str.split(".").str[1].astype(int)

    df["z"] = df["Page"].astype(int)

    return df


def get_remaining_pages(
    pdf: SimpleDirectoryReader, start_page: int, end_page: int
) -> pd.DataFrame:
    """
    Gets the remaining pages of the PDF after a specific range.

    Args:
    pdf (SimpleDirectoryReader): The PDF file.
    start_page (int): The starting page.
    end_page (int): The ending page.

    Returns:
    pd.DataFrame: A dataframe containing the remaining pages.
    """
    remaning_pages = split_pdf(pdf, start_page, end_page)
    # Assuming you've initialized remaning_df somewhere in your code
    remaning_dict = {}
    # Inside your loop
    for page in remaning_pages:
        content = page.text
        page_label = page.metadata["page_label"]
        remaning_dict[page_label] = content
    remaning_df = pd.DataFrame.from_dict(
        remaning_dict, orient="index", columns=["Content"]
    )
    remaning_df = remaning_df.reset_index()
    remaning_df.rename(columns={"index": "Page"}, inplace=True)
    remaning_df["z"] = remaning_df["Page"].astype(int)
    return remaning_df


def remove_patterns(text: str, string_to_remove: str, replacement: str = "") -> str:
    """
    Removes specific patterns from the text.

    Args:
    text (str): The text to process.
    string_to_remove (str): The string pattern to remove.
    replacement (str, optional): The string to replace the pattern with. Defaults to "".

    Returns:
    str: The processed text.
    """
    patterns = [
        r"^\d+\s*\d+\s*\d+\s*\d+\s*\d+\s*[\s\S]*"
        + re.escape(string_to_remove)
        + r"[\s\S]*",
        r"^\d+\s+" + re.escape(string_to_remove),
        r"^\d+\n",
        re.escape(string_to_remove) + r" \d+",
    ]

    for pattern in patterns:
        text = re.sub(pattern, replacement, text)

    return text


def merge_df(
    df: pd.DataFrame, remaning_df: pd.DataFrame, string_to_remove: Optional[str] = None
) -> pd.DataFrame:
    """
    Merges two dataframes and processes them further based on specific conditions.

    Args:
    df (pd.DataFrame): The first dataframe.
    remaning_df (pd.DataFrame): The second dataframe.
    string_to_remove (Optional[str], optional): The string pattern to remove. Defaults to None.

    Returns:
    pd.DataFrame: The merged dataframe.
    """
    merged_df = df.merge(remaning_df, on="Page", how="right")
    df = merged_df.rename(
        columns={"Hierarchy_x": "Hierarchy", "x_x": "x", "y_x": "y", "z_x": "z"}
    )
    df["z"] = df.apply(
        lambda row: row["z_y"] if pd.isna(row["z"]) else row["z"], axis=1
    )
    drop_cols = ["z_y"]
    df = df.drop(drop_cols, axis=1)

    x, y = None, None  # Initialize x and y
    for index, row in df.iterrows():
        if pd.notna(row["Hierarchy"]):
            # If 'Hierarchy' is not NaN, update x and y
            x, y = row["x"], row["y"]
        else:
            # If 'Hierarchy' is NaN, set 'x' and 'y' to the last known values
            df.at[index, "x"] = x
            df.at[index, "y"] = y

    # Fill any remaining NaN values in 'x' and 'y' with appropriate values
    df["x"].fillna(method="ffill", inplace=True)
    df["y"].fillna(method="ffill", inplace=True)

    df = df.drop(["Content_x", "Hierarchy"], axis=1)

    df = df.rename(columns={"Content_y": "Content"})

    df = df.rename(columns={"Page": "page_label"})

    if string_to_remove:
        df["Content"] = df["Content"].apply(remove_patterns, args=(string_to_remove,))

    return df


def proccess_pdf(
    pdf_path: str,
    start_end_page: Tuple[int, int],
    remainig_start_end_page: Tuple[int, int],
    string_to_remove: Optional[str] = None,
    text_column: str = "Content",
) -> SimpleDirectoryReader:
    """
    Processes the PDF file and visualizes data.

    Args:
    pdf_path (str): The path to the PDF file.
    start_end_page (Tuple[int, int]): The start and end page numbers for the main processing.
    remainig_start_end_page (Tuple[int, int]): The start and end page numbers for remaining pages.
    string_to_remove (Optional[str], optional): The string pattern to remove. Defaults to None.
    text_column (str): The column name for text data. Defaults to "Content".
    visualize (bool): Flag to visualize the data. Defaults to True.

    Returns:
    SimpleDirectoryReader: The processed PDF file.
    """
    pdf = get_pdf(pdf_path)
    start_page, end_page = start_end_page
    selected_pages = split_pdf(pdf, start_page, end_page)
    df = append_text(selected_pages)
    chapter_df = format_dataframe(df)
    remainig_start_page, remainig_end_page = remainig_start_end_page

    if remainig_end_page is None:
        remainig_end_page = len(pdf)

    pages_df = get_remaining_pages(pdf, remainig_start_page, remainig_end_page)
    df = merge_df(chapter_df, pages_df, string_to_remove=string_to_remove)

    plot_3d_scatter(
        file_path_or_dataframe=df,
        text_column=text_column,
        show=True,
        label_column="page_label",
    )
    return pdf


def main():
    parser = argparse.ArgumentParser(
        description="Process a PDF file for data visualization."
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="chain_memory/data/CTDAbook.pdf",
        help="Path to the PDF file. Default is 'chain_memory/data/CTDAbook.pdf'.",
    )
    parser.add_argument(
        "--start_page",
        type=int,
        default=1,
        help="Start page for main processing. Default is 4.",
    )
    parser.add_argument(
        "--end_page",
        type=int,
        default=7,
        help="End page for main processing. Default is 9.",
    )
    parser.add_argument(
        "--remaining_start_page",
        type=int,
        default=15,
        help="Start page for remaining pages. Default is 17.",
    )
    parser.add_argument(
        "--remaining_end_page",
        type=int,
        help="End page for remaining pages. If not set, processes till the end of the document.",
    )
    parser.add_argument(
        "--string_to_remove",
        type=str,
        default=None,
        help="String pattern to remove from the content. Default is None.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="Content",
        help="Column name for text data. Default is 'Content'.",
    )

    args = parser.parse_args()

    proccess_pdf(
        pdf_path=args.pdf_path,
        start_end_page=(args.start_page, args.end_page),
        remainig_start_end_page=(args.remaining_start_page, args.remaining_end_page),
        string_to_remove=args.string_to_remove,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
