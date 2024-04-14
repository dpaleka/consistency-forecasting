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


def get_feedback_filename(source_filename):
    return f"{Path(source_filename).stem}_feedback.json"


def write_feedback(entry_id, feedback_data, source_filename):
    feedback_filename = get_feedback_filename(source_filename)
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


def has_previous_feedback(entry_id, source_filename):
    feedback_filename = get_feedback_filename(source_filename)
    feedback_path = Path(source_filename).parent / feedback_filename

    try:
        with open(feedback_path, "r") as file:
            existing_feedback = json.load(file)
    except FileNotFoundError:
        return False

    for feedback_entry in existing_feedback:
        if feedback_entry["question_id"] == entry_id:
            return True

    return False


def get_previous_feedback(entry_id, source_filename):
    feedback_filename = get_feedback_filename(source_filename)
    feedback_path = Path(source_filename).parent / feedback_filename

    try:
        with open(feedback_path, "r") as file:
            existing_feedback = json.load(file)
    except FileNotFoundError:
        return {}

    for feedback_entry in existing_feedback:
        if feedback_entry["question_id"] == entry_id:
            return feedback_entry

    return {}


def display_feedback(feedback):
    st.markdown("### Feedback")

    for field, value in feedback.items():
        st.markdown(f"**{field}:**\n\n{value}\n")


def get_entry(entry_id, entries):
    for entry in entries:
        if entry.get("id") == entry_id:
            return entry
    return None


field_order = [
    "title",
    "body",
    "resolution_date",
    "metadata",
    "resolution",
    "context",
    "id",
]


def display_entry(entry, source_filename, feedback=None):
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
    previous_feedback = has_previous_feedback(entry.get("id", "N/A"), source_filename)

    # Create a layout with two columns
    col1, col2 = st.columns(2)

    # Add the entry details to the first column, don't display empty ones
    with col1:
        for field in field_order:
            if field in entry and entry[field]:
                st.markdown(f"**{field}:**\n\n{entry[field]}\n")
        st.markdown(f"**Has Previous Feedback:**\n\n{previous_feedback}\n")

    # Add the feedback form to the second column
    with col2:
        if feedback:
            display_feedback(feedback)
        else:
            for field in feedback_fields:
                feedback_data[field] = st.text_area(field, "")

            if st.button("Submit Feedback"):
                write_feedback(entry.get("id", "N/A"), feedback_data, source_filename)
                st.success("Feedback submitted successfully!")


def go_back():
    set_view()


def display_list_view(entry):
    previous_feedback = has_previous_feedback(entry.get("id", "N/A"), DEFAULT_FILE)
    st.markdown(f"\n{entry['title']}\n")

    # Create a layout with two columns
    col1, col2 = st.columns(2)

    # Add the "Give feedback" button to the first column
    with col1:
        if st.button(
            "Give feedback",
            on_click=set_view,
            kwargs={"entry": entry},
            key=f"give_feedback{entry.get('id', 'N/A')}",
        ):
            pass

    # Add the "View feedback" button to the second column
    with col2:
        if previous_feedback:
            previous_feedback_data = get_previous_feedback(
                entry.get("id", "N/A"), DEFAULT_FILE
            )
            if st.button(
                "View feedback",
                on_click=set_view,
                kwargs={"feedback": previous_feedback_data, "entry": entry},
                key=f"view_feedback{entry.get('id', 'N/A')}",
            ):
                pass
            st.empty()


def list_view(entries):
    st.title("JSON Lines Viewer")
    for entry in entries:
        display_list_view(entry)


def set_view(entry=None, feedback=None):
    st.session_state.entry_view = entry
    st.session_state.feedback_view = feedback


def main(filename):
    st.set_page_config(layout="wide")
    entries = load_data(filename)

    if "entry_view" not in st.session_state:
        st.session_state.entry_view = None
    if "feedback_view" not in st.session_state:
        st.session_state.feedback_view = None

    if st.session_state.feedback_view:
        display_entry(
            st.session_state.entry_view,
            filename,
            feedback=st.session_state.feedback_view,
        )
        st.button("Back", on_click=go_back)
    elif st.session_state.entry_view:
        display_entry(st.session_state.entry_view, filename)
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
