import re
import xml.etree.ElementTree as ET
from common.datatypes import ResolverOutput
from common.llm_utils import answer
from .prompts import parse_resolver_output_prompt


def parse_xml_resolver_output(full_string: str) -> ResolverOutput:
    try:
        # Extract the XML content using regex
        xml_string = re.search(
            r"<resolver_output>.*?</resolver_output>", full_string, re.DOTALL
        ).group(0)

        # Parse the XML string
        root = ET.fromstring(xml_string)

        # Map XML elements to ResolverOutput model
        return ResolverOutput(
            chain_of_thought=root.findtext("chain_of_thought", "").strip(),
            can_resolve_question=root.findtext("can_resolve_question", "")
            .strip()
            .lower()
            == "true",
            answer=(lambda ans: ans.strip().lower() == "true" if ans else None)(
                root.findtext("answer")
            ),
        )

    except Exception as e:
        raise ValueError(f"Error processing XML: {e}")


async def parse_resolver_output(
    full_string: str, question_title: str, **kwargs
) -> ResolverOutput:
    """
    Example full string:
    <resolver_output>\n  <chain_of_thought>\n    Based on available information:\n    1. The UAW released a new video on August 3, 2024, endorsing Kamala Harris for President.\n    2. This video indicates that the UAW International Executive Board voted to endorse Kamala Harris for President on Wednesday, August 7th, 2024.\n    3. There is no mention of any change in the UAW's endorsement decision after August 2024 in the provided sources.\n\n  </chain_of_thought>\n  <can_resolve_question>true</can_resolve_question>\n  <answer>true</answer>\n</resolver_output>
    """
    try:
        xml_parsed = parse_xml_resolver_output(full_string, **kwargs)
        with open("out_parsed.txt", "a") as f:
            f.write(str(xml_parsed) + "\n")
        return xml_parsed
    except ValueError as e:
        prompt = parse_resolver_output_prompt.format(
            full_string=full_string, question_title=question_title
        )
        with open("out.txt", "a") as f:
            f.write(prompt + "\n")
        r = await answer(prompt, response_model=ResolverOutput, **kwargs)
        return r
    except Exception as e:
        raise ValueError(f"Error parsing resolver output: {e}")
