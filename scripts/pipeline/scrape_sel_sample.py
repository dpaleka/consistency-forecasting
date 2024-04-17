from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException

def fetch_question_details(url):
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    resolution_criteria_text = ""
    background_info_text = ""

    try:
        # Navigate to the page
        driver.get(url)
        time.sleep(1)  # Adjust the sleep time as necessary

        try:
            # Attempt to click "Show More" buttons if present
            show_more_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Show More')]")
            for button in show_more_buttons:
                try:
                    button.click()
                    time.sleep(1)  # Wait for the content to load; adjust as necessary
                except ElementClickInterceptedException:
                    print("Button was not clickable.")
                except:
                    pass
        except NoSuchElementException:
            print("No 'Show more' buttons found.")

        # Wait for the "Resolution Criteria" section to load
        resolution_criteria = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "react-resolution-criteria"))
        )
        resolution_criteria_text = resolution_criteria.text.replace('Show Less', '')

        # Wait for the "Background Info" section to load
        background_info = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "react-background-info"))
        )
        background_info_text = background_info.text.replace('Show Less', '')

    finally:
        # Clean up by closing the browser
        driver.quit()

    return resolution_criteria_text, background_info_text

"""
# Example usage
url = "https://www.metaculus.com/questions/349/spacex-lands-people-on-mars-by-2030/"
resolution_criteria, background_info = fetch_question_details(url)
print("Resolution Criteria:", resolution_criteria)
print("\nBackground Info:", background_info)
"""