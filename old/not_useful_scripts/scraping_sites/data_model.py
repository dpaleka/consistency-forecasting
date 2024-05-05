class PredictionMarketData:
    def __init__(self, website_name, market_id, market_name, market_type, outcomes):
        """
        Initializes a PredictionMarketData instance.
        :param website_name: Name of the website the market data is from.
        :param market_id: Unique identifier for the market.
        :param market_name: Name of the market.
        :param market_type: Type of the market (e.g., binary, multiple-choice).
        :param outcomes: List of possible outcomes for the market.
        """
        self.website_name = website_name
        self.market_id = market_id
        self.market_name = market_name
        self.market_type = market_type
        self.outcomes = outcomes

    def __str__(self):
        """
        String representation of the PredictionMarketData instance.
        """
        return f"Website: {self.website_name}, Market ID: {self.market_id}, Market Name: {self.market_name}, Market Type: {self.market_type}, Outcomes: {self.outcomes}"

    def to_dict(self):
        """
        Converts the PredictionMarketData instance into a dictionary.
        :return: A dictionary representation of the instance.
        """
        return {
            "website_name": self.website_name,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "market_type": self.market_type,
            "outcomes": self.outcomes
        }
