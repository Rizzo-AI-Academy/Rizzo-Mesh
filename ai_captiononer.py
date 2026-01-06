# openai function call to get captionf from an image.
import os
from datapizza.clients.openai import OpenAIClient
from datapizza.clients.openai import OpenAIClient
from datapizza.type import Media, MediaBlock, TextBlock
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAIClient(
    api_key=OPENAI_API_KEY,
    model="gpt-5-mini"  # Vision models required for images
) 
system_prompt="""
Dato il rendering di questo oggetto 3D voglio che mi fornisci una caption breve che lo descriva.
Non indicare il colore dell'oggetto nella descrizione.
"""

def ai_captioning(img_path):
    # Create image media object
    image = Media(
        media_type="image",
        source_type="path",
        source=img_path, # Use the correct path
        extension="jpg"
    )

    # Create media block
    media_block = MediaBlock(media=image)
    text_block = TextBlock(content=system_prompt)

    # Send multimodal input
    response = client.invoke(
        input=[text_block, media_block],
        max_tokens=1024
    )
    return response.text



"""out = ai_captioning("quantized_objs\\0025c5e2333949feb1db259d4ff08dbe_image.jpg")
print(out)"""