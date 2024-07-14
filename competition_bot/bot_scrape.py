import aiohttp


async def fetch_question_details_metaculus(question):
    url = question["metadata"]["api_url"]

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response_json = await response.json()

    resolution_criteria = response_json.get("resolution_criteria", "")
    background_info = response_json.get("description", "")

    community_prediction = response_json.get("community_prediction", {}).get("full", {})
    q2_value = community_prediction.get("q2")
    if q2_value:
        market_prob = q2_value
    else:
        market_prob = None

    question["body"] = resolution_criteria
    question["metadata"]["market_prob"] = market_prob
    question["metadata"]["background_info"] = background_info

    return question
