 # import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
import os

import requests



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


ai_endpoint = secret_client.get_secret("azure-east-us-endpoints").value
ai_key =secret_client.get_secret("azure-east-us-key").value



# Authenticate Azure AI Vision client
cv_client = ImageAnalysisClient(
    endpoint=ai_endpoint,
    credential=AzureKeyCredential(ai_key)
)

image_name = "six.png"

image_url =f"Analyze_image/Images/{image_name}"

with open(image_url, "rb") as img:
    image_data = img.read()


result = cv_client.analyze(
    image_data=image_data,
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.PEOPLE],
)

if result.caption is not None:
    print("\nCaption:")
    print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))



# Get image dense captions
if result.dense_captions is not None:
    print("\nDense Captions:")
    for caption in result.dense_captions.list:
        print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))


# Get image tags
if result.tags is not None:
    print("\nTags:")
    for tag in result.tags.list:
        print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))


if result.objects is not None:
    image = Image.open(image_url)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'Cyan'

    # print(result.objects.list.tags[0].name)   
    print("objects : ")
    for obj_detected in result.objects.list:
        print( obj_detected.tags)
        r = obj_detected.bounding_box
        # print("r : ",r)
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height)) 
        # print("bounding_box: " ,bounding_box)
        draw.rectangle(bounding_box, outline=color, width=3)
        plt.annotate(obj_detected.tags[0].name,(r.x, r.y), backgroundcolor=color) 

    plt.imshow(image)
    plt.tight_layout(pad=0)
    outputfile = f'Analyze_image/Images/test/object_{image_name}'
    fig.savefig(outputfile)
    print('  Results saved in', outputfile)
        # Get people in the image


if result.people is not None:
    print("\nPeople in image:")

    # Prepare image for drawing
    image = Image.open(image_url)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'Cyan'

    for detected_people in result.people.list:
        # Draw object bounding box
        r = detected_people.bounding_box
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
        draw.rectangle(bounding_box, outline="red", width=3)

        # Return the confidence of the person detected
        print(" {} (confidence: {:.2f}%)".format(detected_people.bounding_box, detected_people.confidence * 100))
        
    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    outputfile = f'Analyze_image/Images/test/people_{image_name}'
    fig.savefig(outputfile)
    print('  Results saved in', outputfile)








