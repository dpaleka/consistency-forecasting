from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum, auto


@dataclass
class ForecastingQuestion:
    id: str  # unique id 
    text: str  
    resolution_criteria: str  
    data_source: str  # usually one of “synthetic”, “metaculus”, “manifold”, “predictit”
    question_type: str
    resolution_date: str  
    url: Optional[str] = None
    metadata: Dict[str, List[str]] = field(default_factory=dict)
    resolution: Optional[str] = None  # some questions may have a resolution already

