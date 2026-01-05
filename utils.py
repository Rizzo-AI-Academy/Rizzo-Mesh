import trimesh
import numpy as np
from PIL import Image
import io
import os

RESOLUTION = (1920, 1080)
OBJECT_COLOR = [30, 144, 255, 255]  # Blu

def get_mesh(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"Caricamento di {filepath}...")
        return trimesh.load(filepath)
    else:
        print("Nessun file trovato. Scarico esempio...")
        return trimesh.load_remote('https://github.com/mikedh/trimesh/raw/master/models/featuretype.STL')
    
def render_to_image(mesh, output_file):
    # 1. Flat Shading (Look Low Poly)
    try:
        mesh.unmerge_vertices()
    except:
        pass  # Alcune mesh non supportano unmerge
    
    # 2. Colore
    mesh.visual.face_colors = OBJECT_COLOR
    
    # 3. Scena
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.set_camera(angles=[np.pi/4, np.pi/4, 0], distance=mesh.scale * 2.5)
    
    print("Rendering...")
    
    try:
        png_data = scene.save_image(resolution=RESOLUTION, visible=True)
    except Exception as e:
        print(f"❌ Errore rendering: {e}")
        return

    # 4. Sfondo Bianco
    print("Applicazione sfondo bianco...")
    try:
        image_transparent = Image.open(io.BytesIO(png_data))
        final_image = Image.new("RGB", image_transparent.size, "WHITE")
        final_image.paste(image_transparent, (0, 0), image_transparent)
        final_image.save(output_file, "JPEG")
        print(f"✓ Fatto! Immagine salvata in: {output_file}")
    except Exception as e:
        print(f"❌ Errore salvataggio JPG: {e}")