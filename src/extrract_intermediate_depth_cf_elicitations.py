"""This script is run on consistency evaluation directories
to extract the intermediate CF elicitations, i.e. the ones that are not
at the depth of 4."""


def get_hypocrite_elicitation(line: dict) -> dict:
    hypocrite_lines = {}
    for p in line:
        if p in ["violation_data", "metadata"]:
            continue
        metadata = p["forecast"][""]
        hypocrite_item = [m for m in metadata if m.get("name", None) == "P"]
        assert len(hypocrite_item) == 1, "Expected 1 hypocrite item, got {}".format(
            len(hypocrite_item)
        )
        answer = {
            "prob": hypocrite_item["elicited_prob"],
            "metadata": hypocrite_item["elicitation_metadata"],
        }
        hypocrite_lines[p] = {"question": line[p]["question"], "forecast": answer}
    # hypocrite["violation_data"] = ...
    return hypocrite_lines


def get_intermediate_elicitations(line: dict) -> list[dict]:
    elicitations = []
    while line is not None:
        elicitations.append(get_hypocrite_elicitation(line))
        line = line["metadata"]
    return elicitations
