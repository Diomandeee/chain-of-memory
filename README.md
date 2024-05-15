# 3D Scatter Plot Dash App - Chain Memory

## Description
This Dash application is designed to assess conversation complexity and identify the most optimal messages for fine-tuning chatbot responses. It achieves this by enabling users to visualize and analyze conversational data through an interactive 3D scatter plot. Key features include the ability to classify nodes based on user-defined conditions, offering insights into various aspects of conversation dynamics.

## Features
- **Data Upload**: Users can upload their data in CSV format.
- **Dynamic Filtering**: Apply conditions on different data columns like `depth_x`, `sibling_y`, and `sibling_count_z`.
- **Interactive 3D Visualization**: View the data in an interactive 3D scatter plot.
- **Customizable**: Easy to adapt and extend for various types of datasets and additional features.

## Example Dataset
To help you get started and understand the application's capabilities, I have provided an example conversation dataset. You can find this dataset in the `chain_memory/data` directory. This sample CSV file contains conversational data structured in a format suitable for analysis with our application. It serves as a practical example to demonstrate how the application can be used to assess conversation complexity and optimize chatbot responses.

## Getting Started

These instructions will help you get the application up and running on your local machine for development and testing purposes.

### Setting Up a Virtual Environment

It's recommended to use a virtual environment to manage the dependencies for your project.

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   ```
2. **Activate the Virtual Environment**:
    ```bash
    source venv/bin/activate
    ```

### Installing Dependencies

To install the required packages, run the following command in the root directory of your project:

```bash
pip install -r requirements.txt
```

### Running the Application

To run the app, execute the following command in the root directory of your project:

```bash
python chain_memory/app.py --host 0.0.0.0 --port 8051
```
---

### Running the `process_pdf` Script

The `process_pdf` script is a key component of the Chain Memory Dash App, designed for extracting and visualizing data from PDF documents. 

#### Script Overview
- **PDF Processing**: It handles loading, splitting, and cleaning text from specified pages of a PDF.
- **Data Structuring**: The script creates a hierarchical structure from the text, facilitating a detailed analysis.
- **Visualization Preparation**: Converts the structured data into a DataFrame, ready for 3D scatter plot visualization.
- **Flexibility**: Customizable through command-line arguments to target specific pages, remove unwanted text, and adjust data handling.

#### Running the Script

Run the script in the root directory of your project using the following command:

```bash
python chain_memory/hierarchy.py 
```

Customize the script's execution with arguments like `--pdf_path` for the file location, `--start_page` and `--end_page` for page range, `--remaining_start_page` and `--remaining_end_page` for additional pages, and more. These options allow you to tailor the script to various PDF structures and content needs.

