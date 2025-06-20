import os
import asyncio
import logging
import azure.functions as func
from dotenv import load_dotenv
from azure.identity.aio import ClientSecretCredential
from msgraph import GraphServiceClient

load_dotenv()
SHAREPOINT_HOSTNAME = "chla.sharepoint.com"
SITE_NAME = "patientfamilyeducation"

TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

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
        logging.info(f"Attempting to find the site: '{SITE_NAME}'")
        site = await graph_client.sites.get_by_path(path=f"/sites/{SITE_NAME}", host_name=SHAREPOINT_HOSTNAME)
        if not site or not site.id:
            logging.error("Could not find the SharePoint site. Check SITE_NAME and premissions.")
            return
        logging.info(f"Successfully found the site '{site.display_name}' with ID: {site.id}")
        
        drive = await graph_client.sites.by_site_id(site.id).drive.get()
        if not drive or not drive.id:
            logging.error("Could not find the default document library for the site.")
            return
        logging.info(f"Successfully found the default drive '{drive.name}' with ID: {drive.id}")
        logging.info(f"Listing items in the root of the '{drive.name}' library...")
        children = await graph_client.drives.by_drive_id(drive.id).root.children.get()
        
        if children and children.value:
            for item in children.value:
                item_type = "Folder" if item.folder else "File"
                logging.info(f"Found {item_type}: '{item.name}' (ID: {item.id})")
        else:
            logging.info("No items found in the root of the document library.")
        logging.info("Connection test completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the Graph API call: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(None))