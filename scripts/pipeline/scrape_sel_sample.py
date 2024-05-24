from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time


def fetch_question_details(url):
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

    return resolution_criteria_text, background_info_text


"""
# Example usage
url = "https://www.metaculus.com/questions/349/spacex-lands-people-on-mars-by-2030/"
resolution_criteria, background_info = fetch_question_details(url)
print("Resolution Criteria:", resolution_criteria)
print("\nBackground Info:", background_info)
"""
