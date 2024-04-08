import argparse
import json
import uuid
from pathlib import Path

import streamlit as st


def load_data(filename):
    data = []
    try:
        with open(filename, "r") as file:
            for line in file:
                data.append(json.loads(line))
    except FileNotFoundError:
        st.error("File not found.")
    return data


def write_feedback(entry_id, feedback_data, source_filename):
    feedback_filename = Path(source_filename).stem + "_feedback.json"
    feedback_path = Path(source_filename).parent / feedback_filename

    try:
        with open(feedback_path, "r") as file:
            existing_feedback = json.load(file)
    except FileNotFoundError:
        existing_feedback = []

    feedback_entry = {
        "feedback_id": str(uuid.uuid4()),
        "question_id": entry_id,
        **feedback_data,
    }
    existing_feedback.append(feedback_entry)

    with open(feedback_path, "w") as file:
        json.dump(existing_feedback, file, indent=4)


field_order = [
    "title",
    "body",
    "resolution_date",
    "metadata",
    "resolution",
    "context",
    "id",
]


def display_entry(entry, source_filename):
    st.markdown("### Entry Details")

    feedback_data = {}

    feedback_fields = [
        "Ambiguities",
        "Resolution Criteria",
        "Edge Cases",
        "Relevance of included information",
        "Time Frame",
        "Improved Question wording",
        "Other feedback",
    ]

    # Create a layout with two columns
    col1, col2 = st.columns(2)

    # Add the entry details to the first column, don't display empty ones
    with col1:
        for field in field_order:
            if field in entry and entry[field]:
                st.markdown(f"**{field}:**\n\n{entry[field]}\n")

    # Add the feedback form to the second column
    with col2:
        for field in feedback_fields:
            feedback_data[field] = st.text_area(field, "")

        if st.button("Submit Feedback"):
            write_feedback(entry.get("id", "N/A"), feedback_data, source_filename)
            st.success("Feedback submitted successfully!")


def go_back():
    st.session_state.view = None


def list_view(entries):
    st.title("JSON Lines Viewer")
    for entry in entries:
        if st.button(entry["title"], on_click=set_view, args=(entry,)):
            pass


def set_view(entry):
    st.session_state.view = entry


def main(filename):
    st.set_page_config(layout="wide")
    entries = load_data(filename)

    if "view" not in st.session_state:
        st.session_state.view = None

    if st.session_state.view:
        display_entry(st.session_state.view, filename)
        st.button("Back", on_click=go_back)
    else:
        list_view(entries)


DEFAULT_FILE = "data/politics_qs_1_formated.jsonl"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", default=DEFAULT_FILE, help="Path to the file"
    )
    args = parser.parse_args()
    main(args.filename)
