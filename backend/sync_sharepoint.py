import os
import logging
import msal
import requests
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open('CHLA-RAG-LLM-Capstone-Project-2025/local.settings.json') as f:
        settings = json.load(f)
        config = settings.get("Values", {})
except FileNotFoundError:
    logging.error("FATAL: local.settings.json not found. Please ensure it exists and is properly configured.")
    exit()
    
TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")
SITE_URL = config.get("SITE_URL")
SITE_NAME = config.get("SITE_NAME")
DOC_LIBRARY = config.get("DOC_LIBRARY")

DOWNLOAD_LOCATION = os.path.join(os.path.dirname(__file__), "code", "docs")

# ---- Main Execution ----
def get_access_token():
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.ConfidentialClientApplication(
        CLIENT_ID, authority=authority, client_credential=CLIENT_SECRET
    )
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    
    if "access_token" in result:
        logging.info("Access token acquired successfully")
        return result["access_token"]
    else:
        logging.error("Failed to acquire access token.")
        logging.error(f"Error: {result.get('error')}")
        logging.error(f"Error description: {result.get('error_description')}")
        return None
    
def get_site_id(token):
    graph_url = f"https://graph.microsoft.com/v1.0/sites/{SITE_URL}:/{SITE_NAME}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(graph_url, headers=headers)
    
    if response.status_code == 200:
        site_id = response.json().get("id")
        logging.info(f"Successfully found Site ID: {site_id}")
        return site_id
    else:
        logging.error(f"Error fetching site ID: {response.status_code} - {response.text}")
        return None

def download_all_files_from_library(token, site_id):
    graph_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(graph_url, headers=headers)
    drives = response.json().get("value")
    
    if not drives:
        logging.error("Could not find any drives (document libraries) for the site.")
        return

    drive_id = drives[0].get("id")
    logging.info(f"Found Drive ID: {drive_id}")
    items_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{DOC_LIBRARY}:/children"
    download_count = 0
    
    while items_url:
        response = requests.get(items_url, headers=headers)
        if response.status_code != 200:
            logging.error(f"Error listing files: {response.status_code} - {response.text}")
            break
        
        data = response.json()
        for item in data.get("value", []):
            file_name = item.get('name')
            if file_name and file_name.lower().endswith('.pdf'):
                local_path = os.path.join(DOWNLOAD_LOCATION, file_name)
                if os.path.exists(local_path):
                    logging.info(f"Skipping {file_name}, it already exists")
                    continue
                
                download_url = item.get('@microsoft.graph.downloadUrl')
                if download_url:
                    logging.info(f"Downloading {file_name}...")
                    file_response = requests.get(download_url)
                    with open(local_path, "wb") as f:
                        f.write(file_response.content)
                    download_count += 1
                    
        items_url = data.get("@odata.nextLink")
        
    logging.info(f"Bulk download complete. Downloaded {download_count} new files")
    
if __name__ == "__main__":
    logging.info("Starting the SharePoint syncing process...")
    
    if not os.path.exists(DOWNLOAD_LOCATION):
        os.makedirs(DOWNLOAD_LOCATION)
        logging.info(f"Created download directory: {DOWNLOAD_LOCATION}")
    access_token = get_access_token()
    if access_token:
        site_id = get_site_id(access_token)
        if site_id:
            download_all_files_from_library(access_token, site_id)
    logging.info("SharePoint syncing process complete.")