from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
import os


load_dotenv()
vault_url = os.environ["AZURE_VAULT_KEY"]
client_id = os.environ['AZURE_CLIENT_ID']
tenant_id = os.environ['AZURE_TENANT_ID']
client_secret = os.environ['AZURE_CLIENT_SECRET']


credentials = ClientSecretCredential(
            client_id = client_id, 
            client_secret= client_secret,
            tenant_id= tenant_id
        )

secret_client = SecretClient(vault_url= vault_url, credential= credentials)


ai_endpoint = secret_client.get_secret("ai-endpoints").value
ai_key =secret_client.get_secret("ai-key").value

cv_client = ImageAnalysisClient(
     endpoint=ai_endpoint,
     credential=AzureKeyCredential(ai_key)
 )

for filename in os.listdir('Vision\Extract_text\imgae'):
    image_file = os.path.join('Vision\Extract_text\imgae',filename)

    print(image_file)

    with open(image_file, "rb") as img:
        image_data = img.read()


    result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

    if result.read is not None:
        # print(result.caption)
        # print(result.dense_captions)
        # print(result.read.blocks[0].lines[0].text)


        for i in result.read.blocks:
            for line in i.lines:
                print(line.text)

