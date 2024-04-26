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
    print(f"Data loaded: {data}")  # Print the loaded data for debugging
    return data


def get_feedback_filepath(
    source_filename: str, feedback_abs_dir: str = "src/data/feedback"
):
    source_path = Path(source_filename)
    parts = list(source_path.parts)
    # It's a bit hacky to put all feedback in the same dir no matter what, hope there won't be name conflicts
    if "data" in parts:
        data_index = parts.index("data")
        new_parts = parts[: data_index + 1] + ["feedback"]
    else:
        new_parts = [feedback_abs_dir]

    name = source_path.name.replace(".jsonl", "_feedback.jsonl")
    new_path = Path(*new_parts) / name
    return str(new_path)


def test_get_feedback_filepath():
    assert (
        get_feedback_filepath("data/fq/synthetic/politics_qs_2_formatted.jsonl")
        == "data/feedback/politics_qs_2_formatted_feedback.jsonl"
    )
    assert (
        get_feedback_filepath("src/data/fq/synthetic/politics_qs_2_formatted.jsonl")
        == "src/data/feedback/politics_qs_2_formatted_feedback.jsonl"
    )


test_get_feedback_filepath()


def write_feedback(entry_id, feedback_data, source_filename):
    feedback_path = get_feedback_filepath(source_filename)

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
    feedback_path = get_feedback_filepath(source_filename)
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
    feedback_path = get_feedback_filepath(source_filename)

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
    print(f"Displaying entry details for: {entry}")  # Print the entry details for debugging

    st.markdown("### Entry Details")

    feedback_data = {}

    feedback_fields = {
        "Rewritten body": "Either leave this empty or rewrite the whole body field. This takes precedence over other feedback fields.",
        "Bad included information": "Is there some information irrelevant, time-specific, or is there editorializing? Paste the relevant bit from the body field, and optionally add a comment why it’s bad, and preferably a fix.",
        "Unintuitive/wrong resolution criteria": "Are some items in body unexpected, given the title? Would it be better for downstream consistency checks if the question specified resolution as N/A instead of Yes/No for some edge cases, or vice versa?",
        "Ambiguities": "Specify any ambiguous aspects of the question that could affect its resolution.",
        "Too specific criteria / edge cases": "Are some edge cases extremely low probability, in the sense that it’s clear the question would resolve to N/A if something like this happens?",
        "Edge cases not covered": "Specify any edge cases that the question does not cover but should.",
        "General feedback": "Write anything not covered above.",
        "Formatting issues": "Is some field formatted in an unusual way? Is some field missing?"
    }

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
            # Adjust the conditional logic for displaying examples
            if st.session_state.show_examples:
                for field, description in feedback_fields.items():
                    st.markdown(f"**{field}:**\n\n*Example:* {description}")
                    feedback_data[field] = st.text_area("", "", help=description)
            else:
                for field, description in feedback_fields.items():
                    st.markdown(f"**{field}:**")
                    feedback_data[field] = st.text_area("", "", help=description)

            discard_question = st.radio(
                "Discard the question?",
                ('NO', 'YES'),
                index=0,
                help="Select YES if the question should be discarded, otherwise select NO."
            )

            if discard_question == 'YES':
                feedback_data['Discard reason'] = st.text_area("Reason for discarding the question", "", help="Explain why the question is being discarded.")
            else:
                feedback_data['Discard reason'] = ""

            if st.button("Submit Feedback"):
                write_feedback(entry.get("id", "N/A"), feedback_data, source_filename)
                st.success("Feedback submitted successfully!")


def go_back():
    set_view()


def display_list_view(entry):
    print(f"Displaying entry: {entry.get('id', 'N/A')}")  # Print the entry ID for debugging
    previous_feedback = has_previous_feedback(entry.get("id", "N/A"), DEFAULT_FILE)
    # Use 'text' as the title if 'title' is not present
    entry_title = entry.get('title', entry.get('text', 'No title available'))
    st.markdown(f"\n{entry_title}\n")

    # Create a layout with three columns
    col1, col2, col3 = st.columns(3)

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

    # The "Show/Hide Examples" toggle has been removed from here as per instructions

def list_view(entries):
    print(f"Listing {len(entries)} entries")  # Print the number of entries for debugging
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

    # Move the "Show Examples" checkbox to the top of the main function
    st.session_state.show_examples = st.checkbox("Show Examples")

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


DEFAULT_FILE = "data/fq/synthetic/politics_qs_2_formatted.jsonl"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", default=DEFAULT_FILE, help="Path to the file"
    )
    args, unknown = parser.parse_known_args()  # Ignore unknown args
    main(args.filename)

# Run with:
# streamlit run feedback_form.py -- -f FILENAME.jsonl
