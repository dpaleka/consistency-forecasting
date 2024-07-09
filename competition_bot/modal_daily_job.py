import modal
from pathlib import Path
import subprocess
import os

app = modal.App("daily-metaculus-job")

# Specify the path to your requirements.txt file
bot_path = Path(__file__).parent
root_path = bot_path.parent
env_path = root_path / ".env"
requirements_path = root_path / "requirements.txt"
src_path = root_path / "src"

# Create a Modal image with the requirements installed
env = {
    "NO_CACHE": "True",
    "MAX_CONCURRENT_QUERIES": "20",
    "USE_OPENROUTER": "True",
    "SKIP_NEWSCATCHER": "True",
}

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(requirements_path)
    .pip_install("torch")
    .env(env)
)

# Mount the directories
src_volume = modal.Mount.from_local_dir(src_path, remote_path="/root/src")
bot_volume = modal.Mount.from_local_dir(bot_path, remote_path="/root/competition_bot")


# Create a NetworkFileSystem instance
nfs = modal.NetworkFileSystem.from_name("logs-nfs", create_if_missing=True)

# We pass secrets like this, other relevant env vars above
secrets = [modal.Secret.from_dotenv(env_path)]


@app.function(
    image=image,
    schedule=modal.Period(days=1),
    secrets=secrets,
    mounts=[src_volume, bot_volume, nfs],
    timeout=3 * 3600,  # Increase timeout to 3 hours
)
def run_daily_job():
    os.chdir("/root/competition_bot")
    try:
        # Run the fast script
        process = subprocess.Popen(["python", "metaculus_competition_fast.py"])

        # Wait for 1 hour
        process.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        # If it takes more than 1 hour, terminate the process
        process.terminate()
        process.wait()

        # Run the slow script
        subprocess.run(["python", "metaculus_competition_slow.py"])

    # Copy log files from remote to local
    """
    for log_file in ["metaculus_submissions.log", "metaculus_submission_errors.log"]:
        remote_path = os.path.join("/root/competition_bot", log_file)
        local_path = os.path.join(bot_path, log_file)
        with modal.io.open(remote_path, "rb") as remote_file:
            with open(local_path, "wb") as local_file:
                local_file.write(remote_file.read())
    """


if __name__ == "__main__":
    with app.run():
        run_daily_job.remote()


"""
import modal
from pathlib import Path
import subprocess
import os

app = modal.App("daily-metaculus-job")


# Specify the path to your requirements.txt file
bot_path = Path(__file__).parent

#env_path = bot_path / ".env.bot.example"


root_path = bot_path.parent

env_path = root_path / ".env"

requirements_path = root_path / "requirements.txt"


src_path = root_path / "src"


# Create a Modal image with the requirements installed
env = {
    "NO_CACHE": "True",
    "MAX_CONCURRENT_QUERIES": "20",
    "USE_OPENROUTER": "True",
    "SKIP_NEWSCATCHER": "True",
}

image = (
    modal.Image.debian_slim().pip_install_from_requirements(requirements_path).pip_install("torch")
.env(env)
)

# Mount the directories
src_volume = modal.Mount.from_local_dir(src_path, remote_path="/root/src")
bot_volume = modal.Mount.from_local_dir(bot_path, remote_path="/root/competition_bot")

# We pass secrets like this, other relevant env vars above
secrets = [modal.Secret.from_dotenv(env_path)]



@app.function(
    image=image,
    schedule=modal.Period(days=1),
    secrets=secrets,
    mounts=[src_volume, bot_volume],
)
def run_daily_job():
    os.chdir("/root/competition_bot")
    try:
        # Run the fast script
        process = subprocess.Popen(["python", "metaculus_competition_fast.py"])

        # Wait for 1 hour
        process.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        # If it takes more than 1 hour, terminate the process
        process.terminate()
        process.wait()

        # Run the slow script
        subprocess.run(["python", "metaculus_competition_slow.py"])


if __name__ == "__main__":
    with app.run():
        run_daily_job.remote()

        
"""
