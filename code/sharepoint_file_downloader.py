import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

SHAREPOINT_URL = ""
DOWNLOAD_DIRECTORY = os.path.join(os.getcwd(), "selenium_document_dump")
EDGE_DRIVER_PATH = ""

def create_driver():
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)
        print(f"Created directory: {DOWNLOAD_DIRECTORY}")
    
    edge_options = Options()
    prefs = {"download.default_directory": DOWNLOAD_DIRECTORY}
    edge_options.add_experimental_option("prefs", prefs)
    service = Service(executable_path=EDGE_DRIVER_PATH)
    driver = webdriver.Edge(service=service, options=edge_options)
    return driver

def download_files():
    driver = create_driver()
    driver.get(SHAREPOINT_URL)
    
    print("\n" + "="*50)
    print("ACTION REQUIRED: Please log in to SharePoint in the browser window.")
    print("The script will wait for you. Press Enter in this terminal when you are logged in and on the document library page.")
    print("="*50 + "\n")
    input("Press Enter to continue...")

    print("Login confirmed. Starting the download process...")
    
    page_number = 1
    downloaded_file_count = 0
    
    while True:
        print(f"\n--- Processing Page {page_number} ---")
        try:
            wait = WebDriverWait(driver, 30)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-automationid="DetailsList"]')))
            print("File list container found.")
            time.sleep(5)
            
            file_elements = driver.find_elements(By.CSS_SELECTOR, 'a[data-automationid="ListItem-title"]')
            if not file_elements:
                print("WARNING: No files found on this page with the current selector. The script might end.")
                break
            print(f"Found {len(file_elements)} files on this page.")
            for i in range(len(file_elements)):
                try:
                    current_file = driver.find_elements(By.CSS_SELECTOR, 'a[data-automationid="ListItem-title"]')[i]
                    file_name = current_file.text
                    if os.path.exists(os.path.join(DOWNLOAD_DIRECTORY, file_name)):
                        print(f"  - Skipping {file_name} (already downloaded)")
                        continue
                    print(f"  - Click to download {file_name}")
                    current_file.click()
                    downloaded_file_count += 1
                    time.sleep(3)
                except Exception as e:
                    print(f"  - Error downloading {file_name}: {e}. Skipping file.")
                
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'button[data-automationid="page-button-next"]')
                if next_button.is_enabled():
                    print("Navigating to the next page...")
                    driver.execute_script("arguments[0].click();", next_button)
                    page_number += 1
                    time.sleep(5)
                else:
                    print("'Next' button is disabled. Reached the last page.")
            except NoSuchElementException:
                print("No 'Next' button found. Assuming this is the last page.")
        except TimeoutException:
            print("Timed out waiting for page elements to load. The script will stop.")
            print("This could be due to a wrong URL or incorrect selectors being used.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    
    print("\n" + "="*50)
    print("Download process finished.")
    print(f"Total new files downloaded in this session: {downloaded_file_count}")
    print(f"All files are located in: {DOWNLOAD_DIRECTORY}")
    print("="*50)
    driver.quit()