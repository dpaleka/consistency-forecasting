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
    print(f"Displaying entry details for: {entry}")

    st.markdown("### Entry Details")

    feedback_data = {}
    # fmt: off
    feedback_fields: dict[str, dict[str, str | bool]] = {
        "rewritten_title": {
            "instruction": "New title field.",
            "example": "",
            "always_shown": False,
        },
        "rewritten_body": {
            "instruction": "Write the new body field.",
            "example": "",
            "always_shown": False,
        },
        "rewritten_resolution_date": {
            "instruction": "Write the new resolution date field.",
            "example": "",
            "always_shown": False,
        },
        "bad_or_irrelevant_included_information": {
            "instruction": "Is there some information irrelevant, time-specific, or is there editorializing? Paste the relevant bit from the body field, and optionally add a comment why it’s bad, and preferably a fix.",
            "example": \
                """Example: “As AI continues to evolve, there is growing speculation about its ability to perform complex cognitive tasks that traditionally require human-like understanding and contextual awareness.” is a sentence that adds nothing to the question and should be removed.\n"""  # noqa
                """Example: “Today, Google announced the launch of Google Vids (a new AI-powered video creation app for work).” is not timeless. Just specify what Google Vids was considered to mean at a given date.""",  # noqa
            "always_shown": True,
        },
        "unintuitive_or_wrong_resolution_criteria": {
            "instruction": "Are some items in body unexpected, given the title? Would it be better for downstream consistency checks if the question specified resolution as N/A instead of Yes/No for some edge cases, or vice versa?",
            "example": \
                """Example: “If the 2028 Olympics are canceled, postponed, or otherwise not completed, the question will resolve as No.” should be “will resolve as N/A” for questions dealing with Olympic medal tallies. Otherwise, asking “Will Country X win the most gold medals” over all countries and expecting those to sum up to 1 is not a valid consistency check.\n"""  # noqa
                """Example: “If Italy undergoes significant political or territorial changes before the resolution date that would substantially impact its ability to report emissions or the comparability of emissions data, the question will resolve as No unless a clear and widely accepted method for adjusting the emissions data is provided by an authoritative body.” should be N/A instead.""",  # noqa
            "always_shown": True,
        },
        "too_specific_criteria_or_edge_cases": {
            "instruction": "Are some edge cases extremely low probability, in the sense that it’s clear the question would resolve to N/A if something like this happens?",
            "example": \
                """Example: “If Japan stops existing, the question will resolve as N/A”. """, # noqa
            "always_shown": True,
        },
        "ambiguities": {
            "instruction": "Specify any ambiguous aspects of the question that could affect its resolution.",
            "example": \
                """Example: “Should there be any significant changes to the methodology of how carbon emissions are measured between the time of the question's posting and the resolution date, such changes must be taken into account to ensure a fair assessment of the emissions reduction target.” should be removed and replaced with “If there is a major change in the methodology of how carbon emissions are measured before the resolution date, and it is not possible to measure carbon emissions according to the methodology, this question resolves N/A.”\n"""  # noqa
                """Non-example: Criteria such as “If there are conflicting reports about the fatalities or the nature of military engagement, the question will be resolved by a panel of three experts in international conflict, chosen in good faith by the question author, who will determine whether the criteria have been met based on the preponderance of evidence.” is good if the question cannot specify a trustworthy source for quantitative criteria (such as fatalities in an armed conflict).""",  # noqa
            "always_shown": True,
        },
        "edge_cases_not_covered": {
            "instruction": "Specify any edge cases that the question does not cover but should.",
            "example": \
                """Example: “What is the probability that a Jewish person will be elected...” must include a somewhat precise way to determine if a well-known person is Jewish -- is it religion, or cultural background, self-identification, Rabbinical law, etc? But make it a single sentence, no need to write a full paragraph for this.”\n""",  # noqa
            "always_shown": True,
        },
        "general_feedback": {
            "instruction": "Write anything not covered above.",
            "example": \
                """Example: Too long, shorten""",  # noqa
            "always_shown": True,
        },
        "formatting_issues": {
            "instruction": "Is some field formatted in an unusual way? Is some field missing?",
            "example": \
                """Example: resolution_date: “Resolution date: This question resolves on 28 Jan 2034” instead of “28 Jan 2034” or “28/01/2034”.\n"""  # noqa
                """Example: body: “Resolution criteria: This question will resolve YES if.” instead of just “This question will resolve YES”.""",  # noqa
            "always_shown": True,
        },
        "discard_reason": {
            "instruction": "Explain why the question is being discarded. Make the feedback self-contained in this field only; do not reference what you wrote in other fields.",
            "example": \
                """Example: “What is the probability that the current President/Prime Minister of Spain will be re-elected in the next general election?” depends on when the question is asked.\n"""  # noqa
                """Example: “What is the probability that Shinzo Abe will be re-elected as Prime Minister of Japan in the year 2027?” is off because Shinzo Abe has been assassinated in 2022.""",  # noqa
            "always_shown": False,
        },
    }
    # fmt: on

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
            st.session_state.show_examples = st.checkbox("Show Examples")

            rewrite_title = st.radio(
                "Rewrite title?",
                ("NO", "YES"),
                index=0,
                help="Select YES if you want to rewrite the title, otherwise select NO. This takes precedence over other feedback fields.",
                horizontal=True,
            )
            if rewrite_title == "YES":
                feedback_data["rewritten_title"] = st.text_area(
                    "**title:**",
                    key="rewritten_title",
                    help=feedback_fields["rewritten_title"]["instruction"],
                )
            else:
                feedback_data["rewritten_title"] = ""

            rewrite_body = st.radio(
                "Rewrite body?",
                ("NO", "YES"),
                index=0,
                help="Select YES if you want to rewrite the body, otherwise select NO. This takes precedence over other feedback fields.",
                horizontal=True,
            )
            if rewrite_body == "YES":
                feedback_data["rewritten_body"] = st.text_area(
                    "**body:**",
                    key="rewritten_body",
                    help=feedback_fields["rewritten_body"]["instruction"],
                )
            else:
                feedback_data["rewritten_body"] = ""

            rewrite_resolution_date = st.radio(
                "Rewrite resolution date?",
                ("NO", "YES"),
                index=0,
                help="Select YES if you want to rewrite the resolution date, otherwise select NO. This takes precedence over other feedback fields.",
                horizontal=True,
            )
            if rewrite_resolution_date == "YES":
                feedback_data["rewritten_resolution_date"] = st.text_area(
                    "**resolution_date:**",
                    key="rewritten_resolution_date",
                    help=feedback_fields["rewritten_resolution_date"]["instruction"],
                )
            else:
                feedback_data["rewritten_resolution_date"] = ""

            # Adjust the conditional logic for displaying examples
            for field, data in feedback_fields.items():
                if not data["always_shown"]:
                    continue
                # Check if the 'Show Examples' checkbox is checked
                show_examples = st.session_state.get("show_examples", False)
                print(
                    f"'Show Examples' checkbox state: {show_examples}"
                )  # Debugging print statement
                if show_examples:
                    st.markdown(data["example"])
                # Create a text area for feedback input
                feedback_data[field] = st.text_area(
                    f"**{field}:**",
                    key=f"feedback_{field}",
                    help=data["instruction"],
                )

            discard_question = st.radio(
                "Discard the question?",
                ("NO", "YES"),
                index=0,
                help="Select YES if the question should be discarded, otherwise select NO.",
                horizontal=True,
            )

            if discard_question == "YES":
                feedback_data["discard_reason"] = st.text_area(
                    "**discard_reason:**",
                    key="discard_reason",
                    help="Explain why the question is being discarded.",
                )
            else:
                feedback_data["discard_reason"] = ""

            if st.button("Submit Feedback"):
                write_feedback(entry.get("id", "N/A"), feedback_data, source_filename)
                st.success("Feedback submitted successfully!")


def go_back():
    set_view()


def display_list_view(entry):
    print(
        f"Displaying entry: {entry.get('id', 'N/A')}"
    )  # Print the entry ID for debugging
    previous_feedback = has_previous_feedback(entry.get("id", "N/A"), DEFAULT_FILE)
    # Use 'text' as the title if 'title' is not present
    entry_title = entry.get("title", entry.get("text", "No title available"))
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


def list_view(entries):
    print(
        f"Listing {len(entries)} entries"
    )  # Print the number of entries for debugging
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
