import asyncio
import requests

# from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from arsenic import get_session, keys, browsers, services


chrome_version = ""

if chrome_version == "":
    # Fetch the last known good versions of Chrome for Testing
    response = requests.get(
        "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json"
    )
    data = response.json()
    # Use the version of Chrome for Testing that matches your needs
    # For example, you might want to use the stable version
    chrome_version = data["channels"]["Stable"]["version"]

print(ChromeDriverManager(chrome_version).install())

# Set up the ChromeDriver service to use the specified version
bin_path = ChromeDriverManager(chrome_version).install()
service = services.Chromedriver(binary=bin_path)
browser = browsers.Chrome(capabilities={"args": ["--headless", "--disable-gpu"]})

# browser = browsers.Chrome()


async def hello_world():
    # service = ChromeService(executable_path=ChromeDriverManager(chrome_version).install())
    # browser = browsers.Chrome()
    service = services.Chromedriver()
    browser = browsers.Chrome(chromeOptions={"args": ["--headless", "--disable-gpu"]})
    async with get_session(service, browser) as session:
        async with get_session(service, browser) as session:
            await session.get("https://images.google.com/")
            search_box = await session.wait_for_element(5, "input[name=q]")
            await search_box.send_keys("Cats")
            await search_box.send_keys(keys.ENTER)
            await asyncio.sleep(10)


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(hello_world())


if __name__ == "__main__":
    main()
