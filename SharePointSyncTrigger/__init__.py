import os
import asyncio
import logging
import azure.functions as func
import requests
from dotenv import load_dotenv
from azure.identity.aio import ClientSecretCredential
from msgraph import GraphServiceClient

load_dotenv()
SHAREPOINT_HOSTNAME = "chla.sharepoint.com"
SITE_NAME = "patientfamilyeducation"

TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

output_dir = "docs"
os.makedirs(output_dir, exist_ok=True)

async def main(mytimer):
    logging.info("SharePoint connection test function triggered.")
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    scopes = ["https://graph.microsoft.com/.default"]
    graph_client = GraphServiceClient(credentials=credential, scopes=scopes)
    
    try:
        # 1. Get the SharePoint site ID
        logging.info(f"Attempting to find the site: '{SITE_NAME}'")
        site = await graph_client.sites.get_by_path(path=f"/sites/{SITE_NAME}", host_name=SHAREPOINT_HOSTNAME)
        if not site or not site.id:
            logging.error("Could not find the SharePoint site. Check SITE_NAME and premissions.")
            return
        logging.info(f"Successfully found the site '{site.display_name}' with ID: {site.id}")
        
        # 2. Get the default Document Library (Drive) ID
        drive = await graph_client.sites.by_site_id(site.id).drive.get()
        if not drive or not drive.id:
            logging.error("Could not find the default document library for the site.")
            return
        logging.info(f"Successfully found the default drive '{drive.name}' with ID: {drive.id}")
        
        # 3. List the items and find the first file
        logging.info(f"Listing items in the root of the '{drive.name}' library...")
        children = await graph_client.drives.by_drive_id(drive.id).root.children.get()
        
        if children and children.value:
            for item in children.value:
                if item.file:
                    logging.info(f"Found a file: '{item.name}")
                    download_url = item.additional_data.get('@microsoft.graph.downloadUrl')
                    if download_url:
                        logging.info(f"Downloading '{item.name} from {download_url}")
                        response = requests.get(download_url)
                        response.raise_for_status()
                        file_path = os.path.join(output_dir, item.name)
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        logging.info(f"Successfully saved '{item.name}' to your computer.")
                        break
        else:
            logging.info("No items found in the root of the document library.")
            
        logging.info("Download test completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the Graph API call: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(None))