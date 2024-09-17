import requests
import asyncio
import os
from dotenv import load_dotenv
import aiohttp
import json


class BasePerplexityClient:
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://api.perplexity.ai"
        self.chat = type("ChatNamespace", (), {})()
        self.chat.completions = self.ChatCompletions(self)

    class ChatCompletions:
        def __init__(self, client):
            self.client = client

        def _process_model_name(self, model):
            if model.startswith("perplexity/"):
                return model.split("/", 1)[1]
            return model

        def _prepare_payload(self, model, messages, **kwargs):
            processed_model = self._process_model_name(model)
            return {
                "model": processed_model,
                "messages": messages,
                "return_citations": True,
                **kwargs,
            }

        def _prepare_headers(self):
            return {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.client.api_token}",
            }


class SyncPerplexityClient(BasePerplexityClient):
    class ChatCompletions(BasePerplexityClient.ChatCompletions):
        def create_with_completion(
            self, messages, model="llama-3.1-sonar-large-128k-online", **kwargs
        ):
            url = f"{self.client.base_url}/chat/completions"
            payload = self._prepare_payload(model, messages, **kwargs)
            headers = self._prepare_headers()
            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"


class AsyncPerplexityClient(BasePerplexityClient):
    class ChatCompletions(BasePerplexityClient.ChatCompletions):
        async def create_with_completion(
            self, messages, model="llama-3.1-sonar-large-128k-online", **kwargs
        ):
            url = f"{self.client.base_url}/chat/completions"
            payload = self._prepare_payload(model, messages, **kwargs)
            headers = self._prepare_headers()
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        return f"Error: {response.status} - {await response.text()}"


# Usage examples:
# Synchronous client
# sync_client = SyncPerplexityClient("your_api_token")
# result = sync_client.chat.completions.create_with_completion([
#     {"role": "system", "content": "Be precise and concise."},
#     {"role": "user", "content": "What is the capital of France?"}
# ])
# print(result)


def get_api_token():
    load_dotenv()  # Load environment variables from .env file
    api_token = os.getenv("PERPLEXITY_API_KEY")
    if not api_token:
        raise ValueError("PERPLEXITY_API_TOKEN not found in .env file")
    return api_token


async def main():
    api_token = get_api_token()
    async_client = AsyncPerplexityClient(api_token)
    result = await async_client.chat.completions.create_with_completion(
        [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
