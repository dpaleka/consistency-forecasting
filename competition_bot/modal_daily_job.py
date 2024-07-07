import modal
from pathlib import Path
import subprocess
import os

stub = modal.App("daily-metaculus-job")


# Specify the path to your requirements.txt file
bot_path = Path(__file__).parent

env_path = bot_path / ".env.bot"

root_path = bot_path.parent
requirements_path = root_path / "requirements.txt"
src_path = root_path / "src"

# Create a Modal image with the requirements installed
image = modal.Image.debian_slim().pip_install_from_requirements(requirements_path)

# Mount the src directory
src_volume = modal.Mount.from_local_dir(src_path, remote_path="/root/src")
bot_volume = modal.Mount.from_local_dir(bot_path, remote_path="/root/competition_bot")
secrets = [modal.Secret.from_dotenv(env_path)]


@stub.function(
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
    with stub.run():
        run_daily_job.remote()
