import streamlit as st
import json
import uuid
from pathlib import Path

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
        **feedback_data
    }
    existing_feedback.append(feedback_entry)

    with open(feedback_path, "w") as file:
        json.dump(existing_feedback, file, indent=4)

def display_entry(entry, source_filename):
    st.markdown("### Entry Details")
    for key, value in entry.items():
        st.markdown(f"**{key}:**\n\n{value}\n")

    st.markdown("---")
    st.markdown("### Provide Feedback")

    feedback_data = {}

    feedback_fields = [
        "Ambiguities", 
        "Resolution Criteria", 
        "Edge Cases", 
        "Relevance of included information", 
        "Time Frame", 
        "Improved Question wording", 
        "Other feedback"
    ]
    
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

def main():
    filename = "data/politics_qs_1_formated.jsonl"
    entries = load_data(filename)
    
    if "view" not in st.session_state:
        st.session_state.view = None

    if st.session_state.view:
        display_entry(st.session_state.view, filename)
        st.button("Back", on_click=go_back)
    else:
        list_view(entries)

if __name__ == "__main__":
    main()
