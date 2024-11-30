## Bot for the Metaculus competition
We run the forecasters on the Metaculus [AI Forecasting Benchmark Series (July 2024)](https://www.metaculus.com/notebooks/25525/announcing-the-ai-forecasting-benchmark-series--july-8-120k-in-prizes/) (contact @amaster97 for details).
The bot to call the forecasters is developed in `src/metaculus_competition_fast.py`.  If that doesn't "work" due to issues with concurrency, a similar fallback script `src/metaculus_competition_slow.py`
 can be substituted.  It can be run as a single job on [Modal](https://modal.com/) by doing:
```
python competition_bot/modal_daily_job.py
```

Furthermore, modal also allows the deployment of timed runs for the program (in this case daily).  To deploy a recurring job use:

```
modal deploy competition_bot/modal_daily_job.py
```


To set up, you need Modal credentials.
Do not try to install `modal` in the main Python environment.
Instead, make a new virtual environment and install the requirements with:
```
pip install -r competition_bot/modal_requirements.txt
modal token new
```
and follow the instructions.

In addition to the LLM API costs, each daily run costs Modal credits for the CPU time occupied. Modal gives $30 in credits to new users, and it should be enough for this competition.

### Logs
We store logs of both the question - submissions as well as any potential errors.
In the modal app we have these stored at:
LOG_FILE_PATH = "/mnt/logs/metaculus_submissions.log"
ERROR_LOG_FILE_PATH = "/mnt/logs/metaculus_submission_errors.log"