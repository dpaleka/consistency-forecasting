# Local application/library specific imports
from prompts.prompts import PROMPT_DICT

OAI_SOURCE = "OAI"
ANTHROPIC_SOURCE = "ANTHROPIC"
TOGETHER_AI_SOURCE = "TOGETHER"
GOOGLE_SOURCE = "GOOGLE"
HUGGINGFACE_SOURCE = "HUGGINGFACE"

CHARS_PER_TOKEN = 4
ANTHROPIC_RATE_LIMIT = 5

DEFAULT_RETRIEVAL_CONFIG = {
    "NUM_SEARCH_QUERY_KEYWORDS": 3,
    "MAX_WORDS_NEWSCATCHER": 5,
    "MAX_WORDS_GNEWS": 8,
    "SEARCH_QUERY_MODEL_NAME": "gpt-4-1106-preview",
    "SEARCH_QUERY_TEMPERATURE": 0.0,
    "SEARCH_QUERY_PROMPT_TEMPLATES": [
        PROMPT_DICT["search_query"]["0"],
        PROMPT_DICT["search_query"]["1"],
    ],
    "NUM_ARTICLES_PER_QUERY": 5,
    "SUMMARIZATION_MODEL_NAME": "gpt-4o-mini-2024-07-18",
    "SUMMARIZATION_TEMPERATURE": 0.2,
    "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
    "PRE_FILTER_WITH_EMBEDDING": True,
    "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
    "RANKING_MODEL_NAME": "gpt-4o-mini-2024-07-18",
    "RANKING_TEMPERATURE": 0.0,
    "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
    "RANKING_RELEVANCE_THRESHOLD": 4,
    "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
    "SORT_BY": "date",
    "RANKING_METHOD": "llm-rating",
    "RANKING_METHOD_LLM": "title_250_tokens",
    "NUM_SUMMARIES_THRESHOLD": 20,
    "EXTRACT_BACKGROUND_URLS": True,
}

DEFAULT_REASONING_CONFIG = {
    "BASE_REASONING_MODEL_NAMES": ["gpt-4-1106-preview"],
    "BASE_REASONING_TEMPERATURE": 1.0,
    "BASE_REASONING_PROMPT_TEMPLATES": [
        [
            PROMPT_DICT["binary"]["scratch_pad"]["1"],
            PROMPT_DICT["binary"]["scratch_pad"]["2"],
        ],
    ],
    "AGGREGATION_METHOD": "meta",
    "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
    "AGGREGATION_TEMPERATURE": 0.2,
    "AGGREGATION_MODEL_NAME": "gpt-4",
    "AGGREGATION_WEIGHTS": None,
}


MODEL_TOKEN_LIMITS = {
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-3-haiku-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-opus-20240229": 200000,
    "anthropic/claude-3-haiku": 200000,
    "anthropic/claude-3-sonnet": 200000,
    "anthropic/claude-3-opus": 200000,
    "gpt-4": 8000,
    "gpt-3.5-turbo-1106": 16000,
    "gpt-3.5-turbo-0125": 16000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-3.5-turbo": 16000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4o-mini": 128000,
    "gemini-pro": 30720,
    "togethercomputer/llama-2-7b-chat": 4096,
    "togethercomputer/llama-2-13b-chat": 4096,
    "togethercomputer/llama-2-70b-chat": 4096,
    "togethercomputer/StripedHyena-Hessian-7B": 32768,
    "togethercomputer/LLaMA-2-7B-32K": 32768,
    "mistralai/Mistral-7B-Instruct-v0.2": 32768,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "zero-one-ai/Yi-34B-Chat": 4096,
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 32768,
    "NousResearch/Nous-Hermes-2-Yi-34B": 32768,
}


IRRETRIEVABLE_SITES = [
    "wsj.com",
    "english.alarabiya.net",
    "consilium.europa.eu",
    "abc.net.au",
    "thehill.com",
    "democracynow.org",
    "fifa.com",
    "si.com",
    "aa.com.tr",
    "thestreet.com",
    "newsweek.com",
    "spokesman.com",
    "aninews.in",
    "commonslibrary.parliament.uk",
    "cybernews.com",
    "lineups.com",
    "expressnews.com",
    "news-herald.com",
    "c-span.org/video",
    "investors.com",
    "finance.yahoo.com",  # This site has a “read more” button.
    "metaculus.com",  # newspaper4k cannot parse metaculus pages well
    "houstonchronicle.com",
    "unrwa.org",
    "njspotlightnews.org",
    "crisisgroup.org",
    "vanguardngr.com",  # protected by Cloudflare
    "ahram.org.eg",  # protected by Cloudflare
    "reuters.com",  # blocked by Javascript and CAPTCHA
    "carnegieendowment.org",
    "casino.org",
    "legalsportsreport.com",
    "thehockeynews.com",
    "yna.co.kr",
    "carrefour.com",
    "carnegieeurope.eu",
    "arabianbusiness.com",
    "inc.com",
    "joburg.org.za",
    "timesofindia.indiatimes.com",
    "seekingalpha.com",
    "producer.com",  # protected by Cloudflare
    "oecd.org",
    "almayadeen.net",  # protected by Cloudflare
    "manifold.markets",  # prevent data contamination
    "goodjudgment.com",  # prevent data contamination
    "infer-pub.com",  # prevent data contamination
    "www.gjopen.com",  # prevent data contamination
    "polymarket.com",  # prevent data contamination
    "betting.betfair.com",  # protected by Cloudflare
    "news.com.au",  # blocks crawler
    "predictit.org",  # prevent data contamination
    "atozsports.com",
    "barrons.com",
    "forex.com",
    "www.cnbc.com/quotes",  # stock market data: prevent data contamination
    "montrealgazette.com",
    "bangkokpost.com",
    "editorandpublisher.com",
    "realcleardefense.com",
    "axios.com",
    "mensjournal.com",
    "warriormaven.com",
    "tapinto.net",
    "indianexpress.com",
    "science.org",
    "businessdesk.co.nz",
    "mmanews.com",
    "jdpower.com",
    "hrexchangenetwork.com",
    "arabnews.com",
    "nationalpost.com",
    "bizjournals.com",
    "thejakartapost.com",
]
