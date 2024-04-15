"""
Format 1 (example):
    {
        "id": 349,
        "title": "Will SpaceX land people on Mars before 2030?",
        "body": {
            "resolution_criteria": "Resolution Criteria\nThis question will resolve as Yes if a SpaceX-branded mission successfully lands one or more living human beings on the surface of Mars before 2030. The landing itself of the human crew on Mars must occur before January 1, 2030, 00:00 UTC.\nAt least one person aboard the lander must survive the landing, however it is not necessary for the person to survive long-term or make a return trip to Earth, nor is it necessary for the mission to intend a return or long-term survival.\nA \"SpaceX-branded\" mission is defined to mean that the SpaceX-associated logos on the spacecraft involved (both the boosters and the Mars-bound craft) have a larger surface area than the logos of any other entity\n",
            "background_info": "SpaceX recently released a detailed plan (transcription and slides here) to send people to Mars using an \"Interplanetary Transport System\" based on heavily reusable launch boosters, tanker-assisted refueling in low-Earth orbit, and a futuristic interplanetary spaceship. The ship is to traverse deep space and land intact on Mars after a high-speed retro-assisted atmospheric entry. The system will rely on in-situ fuel generation on Mars for return journeys, and it is envisioned that destinations across the Solar System may be within its reach.\nThe timeline has not been set in stone, but Elon Musk has noted that if SpaceX \"gets lucky and things go according to plan\", a manned flight could launch in the 2024 window with a landing on Mars in 2025. Subsequent launch windows, which are dictated by the Earth-Mars synodic period, occur at a roughly 2-year cadence.\nThere have been numerous proposals over the years for landing people on Mars. Perhaps the first one that was both concrete and marginally credible was Wernher von Braun's Marsprojekt of the late 1940s and early 1950s. For the past six decades, trips to Mars have tended to lie 20-30 years in the future. The SpaceX plan is particularly notable for aggressively compressing the timeline.\n"
        },
        "question_type": "binary",
        "resolution_date": "Jan 02 2030 12:00AM",
        "url": "https://www.metaculus.com/questions/349",
        "data_source": "metaculus",
        "metadata": {
            "topics": [
                {
                    "id": 184,
                    "slug": "elon-musk",
                    "name": "Elon Musk",
                    "link_id": 27681,
                    "num_questions": 155
                },
                {
                    "id": 485,
                    "slug": "spacex-reusable-launch-system-development-program",
                    "name": "SpaceX reusable launch system",
                    "link_id": 27682,
                    "num_questions": 130
                },
                {
                    "id": 1365,
                    "slug": "spacex",
                    "name": "SpaceX",
                    "link_id": 75197,
                    "num_questions": 110
                },
                {
                    "id": 564,
                    "slug": "colonization-of-mars",
                    "name": "Colonization of Mars",
                    "link_id": 27683,
                    "num_questions": 70
                },
                {
                    "id": 1768,
                    "slug": "spacex-mars-transportation-infrastructure",
                    "name": "SpaceX Mars transportation infrastructure",
                    "link_id": 40982,
                    "num_questions": 5
                }
            ]
        },
        "resolution": null
    },
"""

"""
Format 2 (example):
{"id": "e1fe03a9-d89c-43a4-8d18-c9f160c27801", "title": "What is the probability that the United Kingdom will meet its target of reducing greenhouse gas emissions by 68% by 2030?", "body": "Resolution criteria: This question will resolve as Yes if the United Kingdom achieves a reduction in greenhouse gas emissions by 68% or more by December 31, 2030, compared to 1990 levels. The resolution will be based on the official data published by the UK's Department for Business, Energy & Industrial Strategy (BEIS) or its successor agency. The data must be the final confirmed statistics for the year 2030, not preliminary or projected figures. If the BEIS is dissolved or no longer publishes this data, the resolution will be based on the official statistics published by the United Nations Framework Convention on Climate Change (UNFCCC) or its successor agency.\n\nIn the event that neither the BEIS nor the UNFCCC publishes the relevant data by the resolution date, the question will resolve using the most recent data available from either of these sources, provided that the data covers the entirety of 2030.\n\nIf the UK undergoes significant changes in territory or sovereignty that would materially impact its ability to meet the emissions target (such as a secession of a part of the country), the question will resolve based on the emissions of the territory as defined at the time of the question's posting.\n\nIf the methodology for calculating greenhouse gas emissions is significantly altered from the one used as of 2021, the resolution will be based on the best available method that is consistent with the original methodology to the extent possible.\n\nEdge cases, such as significant natural disasters or other extraordinary events that could temporarily skew emissions data, will be considered by a panel of three climate policy experts chosen in good faith by the question author, for the sole purpose of resolving this question.\n\nResolution date: The resolution date will be set to allow for the official data to be published and confirmed. As such, the resolution date will be six months after the end of the target year to account for reporting and verification processes.\n\n", "data_source": "synthetic", "question_type": "Binary", "resolution_date": "\nResolution date: 30/06/2031", "url": null, "metadata": {}, "resolution": null}
"""

def convert_format(format_1):
    format_2 = {
        "id": str(format_1["id"]),
        "title": format_1["title"],
        "body": format_1["body"],
        "data_source": format_1["data_source"],
        "question_type": format_1["question_type"].lower(),
        "resolution_date": format_1["resolution_date"],
        "url": format_1.get("url", None),
        "metadata": format_1["metadata"],
        "resolution": None
    }
    # Fix body into resolution criteria and metadata. 
    if isinstance(format_2["body"], dict) and "resolution_criteria" in format_2["body"]:
        format_2["metadata"]["background_info"] = format_2["body"]["background_info"]
        format_2["body"] = format_2["body"]["resolution_criteria"]
        assert isinstance(format_2["body"], str)
    

    return format_2


from dataclasses import dataclass
from simple_parsing import ArgumentParser
import json
parser = ArgumentParser()

@dataclass
class Options:
    filename: str       # Filename to apply the conversion to

args = parser.add_arguments(Options, dest="opt")
args = parser.parse_args()
args = args.opt
print(type(args), dir(args))

with open(args.filename, "r") as f:
    if args.filename.endswith('.json'):
        data = json.load(f)
    elif args.filename.endswith('.jsonl'):
        data = [json.loads(line) for line in f]

# write jsonl
out_filename = args.filename.replace('.json', '.jsonl')
with open(out_filename, "w") as f:
    for datum in data:
        f.write(json.dumps(convert_format(datum)) + "\n")

