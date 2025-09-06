# GOTYOU
import base64
from pathlib import Path
from langchain_core.tools import tool
import os
def load_image(filename: str) -> str:
    local_img = Path(filename)
    image_data = base64.b64encode(local_img.read_bytes()).decode("utf-8")

    content =  [
            {
                "type": "text",
                "text": "Extract the table from this image output as markdown table.",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/jpeg",
            },
        ]
    return content

@tool
def save_markdown_table(table: str, filename: str) -> None:
    """Save a markdown table to a file."""
    try:
        filename = filename.strip()
        folder = os.environ.get("OUTPUT_FOLDER","/home/victor/workspace/playgrounds/langchain/test_data/image_table")
        output = Path(folder)/filename
        with output.open("w") as f:
            f.write(table)
        return f"Saved markdown table to {output}"
    except Exception as e:
        return f"Error saving markdown table {output}: {e}"