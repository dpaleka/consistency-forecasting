"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
"""


import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re

from playwright.async_api import async_playwright


async def get_market_prob(question):
    pass


def fetch_question_details_predictit(question):
    pass


async def fetch_question_details_metaculus(question):
    url = question["url"]
    print(url)
    resolution_criteria_text = ""
    background_info_text = ""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url)

        question["metadata"]["market_prob"] = None
        if question["resolution"] is None:
            page_content = await page.content()

            # Search for the pattern
            prob_match = re.search(
                r"Metaculus community forecasters was (\d+)%", page_content
            )

            if prob_match:
                prob = 0.01 * float(prob_match.group(1))
                question["metadata"]["market_prob"] = prob

        # Find all "Show More" buttons. Adjust the selector as needed.
        # show_more_buttons = await page.query_selector_all("//button[contains(text(), 'Show More')]")
        show_more_buttons = await page.query_selector_all(
            "//button[text()='Show More']"
        )

        for index, button in enumerate(show_more_buttons):
            if index > 1:
                break
            try:
                # Scroll to each button and click it
                await asyncio.wait_for(button.scroll_into_view_if_needed(), timeout=15)
                await asyncio.sleep(1)  # Allow time for any dynamic content to load

                # Scroll up a little to adjust for fixed headers or other elements
                await asyncio.wait_for(
                    page.evaluate("window.scrollBy(0, -100);"), timeout=15
                )
                await asyncio.sleep(1)  # Allow time for any dynamic content to load

                await asyncio.wait_for(button.click(), timeout=15)
                await asyncio.sleep(1)  # Allow time for any dynamic content to load

            except asyncio.TimeoutError:
                print(
                    f"Operation timed out for button at index {index} at {url}. Moving to the next button."
                )
                continue

        try:
            resolution_criteria = await page.query_selector("react-resolution-criteria")
            resolution_criteria_text = await resolution_criteria.text_content()
            resolution_criteria_text = resolution_criteria_text.replace("Show Less", "")

            background_info = await page.query_selector("react-background-info")
            background_info_text = await background_info.text_content()
            background_info_text = background_info_text.replace("Show Less", "")
        except asyncio.TimeoutError:
            print(
                f"Operation timed out for button at for writing info at {url}. Moving to the next url."
            )

        await browser.close()

    question["body"] = {
        "resolution_criteria": resolution_criteria_text,
        "background_info": background_info_text,
    }
    return question


async def fetch_question_details_manifold(question):
    url = question["url"]
    resolution_criteria_text = ""
    background_info_text = ""

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

            meta_tag = soup.find("meta", attrs={"name": "description"})

            ##Removes the extra xx% chance if it exists
            content = meta_tag["content"] if meta_tag else ""

        chance_match = re.search(r"(\d+)% chance", content)

        if chance_match:
            prob = (
                float(chance_match.group(1)) * 0.01
            )  # Convert the matched string to an integer
            question["metadata"]["market_prob"] = prob
            content = re.sub(r"^\d+% chance\. ", "", content)
        else:
            question["metadata"]["market_prob"] = prob

        ## Make LLM divide up into resolution criteria or background text

        background_info_text = content
        resolution_criteria_text = "nini"

        """
        msgs = 


        async def query_api_chat(
            messages: list[dict[str, str]],
            verbose=False,
            model: str | None = None,
            **kwargs,
        ) -> BaseModel:       

        """

        question["body"] = {
            "resolution_criteria": resolution_criteria_text,
            "background_info": background_info_text,
        }
        return question


"""
def fetch_question_details_metaculus(question):
    url = question['url']
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    resolution_criteria_text = ""
    background_info_text = ""

    try:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Find all "Show More" buttons. Adjust the selector as needed.
        show_more_buttons = driver.find_elements(
            By.XPATH, "//button[contains(text(), 'Show More')]"
        )

        for index, button in enumerate(show_more_buttons):
            if index > 1:
                break
            # Scroll to each button and click it
            driver.execute_script("arguments[0].scrollIntoView(true);", button)
            time.sleep(1)  # Adjust timing as necessary

            # Scroll up a little to adjust for fixed headers or other elements
            driver.execute_script(
                "window.scrollBy(0, -100);"
            )  # Adjust the value as needed
            time.sleep(
                1
            )  # Allow time for any dynamic content to load and for the page to settle

            attempts = 5
            for i in range(attempts):
                try:
                    button.click()
                    time.sleep(1)  # Allow time for any dynamic content to load
                    break
                except Exception as e:
                    print(f"Could not click the button: {e}")
                    i += 1

        resolution_criteria = WebDriverWait(driver, 1.5).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "react-resolution-criteria")
            )
        )
        resolution_criteria_text = resolution_criteria.text.replace("Show Less", "")

        background_info = WebDriverWait(driver, 1.5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "react-background-info"))
        )
        background_info_text = background_info.text.replace("Show Less", "")

    finally:
        driver.quit()

    question['body'] = {
        "resolution_criteria": resolution_criteria_text,
        "background_info": background_info_text
    }
    
    return question
"""


"""
# Example usage
url = "https://www.metaculus.com/questions/349/spacex-lands-people-on-mars-by-2030/"
resolution_criteria, background_info = fetch_question_details(url)
print("Resolution Criteria:", resolution_criteria)
print("\nBackground Info:", background_info)
"""
