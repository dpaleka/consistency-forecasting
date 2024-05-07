# Necessary to make this import work, if not installing it as a submodule
import sys
from pathlib import Path


def get_llm_forecasting_path() -> Path:
    # from common.path_utils import get_src_path
    # sys.path.append(str(get_src_path() / "forecasters/llm_forecasting"))
    return Path(__file__).parent.resolve()


sys.path.append(str(get_llm_forecasting_path()))
