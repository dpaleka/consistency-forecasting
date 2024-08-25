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
    "NO_CACHE": "true",
    "MAX_CONCURRENT_QUERIES": "20",
    # "USE_OPENROUTER": "false",
    "SKIP_NEWSCATCHER": "true",
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
volume = modal.Volume.from_name("logs-volume", create_if_missing=True)

# We pass secrets like this, other relevant env vars above
secrets = [modal.Secret.from_dotenv(env_path)]


@app.function(
    image=image,
    schedule=modal.Period(days=1),
    secrets=secrets,
    mounts=[src_volume, bot_volume],
    volumes={"/mnt/logs": volume},  # Mount the volume at /mnt/logs
    timeout=3 * 3600,  # Increase timeout to 3 hours
)
def run_daily_job():
    os.chdir("/root/competition_bot")
    try:
        # Run the fast script
        process = subprocess.Popen(["python", "metaculus_competition_fast.py"])

        volume.commit()
        process.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        # If it takes more than 1 hour, terminate the process
        process.terminate()
        process.wait()

        # Run the slow script
        subprocess.run(["python", "metaculus_competition_slow.py"])

        # Commit changes to the volume
        volume.commit()


if __name__ == "__main__":
    with app.run():
        run_daily_job.remote()
