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
    try:
        return parse_xml_resolver_output(full_string, **kwargs)
    except ValueError as e:
        prompt = parse_resolver_output_prompt.format(
            full_string=full_string, question_title=question_title
        )
        r = await answer(prompt, response_model=ResolverOutput, **kwargs)
        return r
