from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import matplotlib.pyplot as plt

from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
import os



from PIL import Image, ImageDraw

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
    file_name = filename
    image_file = os.path.join('Vision\Extract_text\imgae',filename)

    print(image_file)
    with open(image_file, "rb") as img:
        image_data = img.read()


    result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )
    
    if result.read is not None:


        with Image.open(image_file) as image:
            fig = plt.figure(figsize=(image.width/100, image.height/100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'Green'


            for line in result.read.blocks[0].lines:
                print(f"  {line.text}") 
                drawLinePloygon = True
                r = line.bounding_polygon
                bounding_polygon = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
                # print("Bounding_Polygon :", bounding_polygon)

                for word in  line.words:
                    rt = word.bounding_polygon
                    bounding_polygon1 = ((rt[0].x,rt[0].y), (rt[1].x, rt[1].y), (rt[2].x, rt[2].y), (rt[3].x, rt[3].y))
                    # print(f"    Word: '{word.text}', Bounding Polygon: {bounding_polygon}, Confidence: {word.confidence:.4f}")
                    drawLinePolygon = False
                    draw.polygon(bounding_polygon1, outline=color, width=5)

                if drawLinePolygon:
                    draw.polygon(bounding_polygon, outline="red", width=3)
            # Save image
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = "Vision/Extract_text/imgae/test/" + file_name
            fig.savefig(outputfile)
            print('\n  Results saved in', outputfile)       







        
