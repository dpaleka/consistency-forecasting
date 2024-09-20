from forecasters.forecaster import Forecaster
from forecasters.forecaster import Forecast


class DummyForecaster(Forecaster):
    def __init__(self, model: str, test_option: list):
        self.model = model
        self.test_option: list = test_option

    def call(self, fq, **kwargs):
        return Forecast(prob=0.5, metadata=None)

    async def call_async(self, fq, **kwargs):
        return Forecast(prob=0.5, metadata=None)

    def dump_config(self):
        return {"model": self.model, "test_option": self.test_option}

    @classmethod
    def load_config(cls, config):
        return cls(model=config["model"], test_option=config["test_option"])
