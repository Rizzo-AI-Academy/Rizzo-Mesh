import trimesh
import numpy as np
from PIL import Image
import io
import os

RESOLUTION = (1920, 1080)
OBJECT_COLOR = [30, 144, 255, 255]  # Blu

def get_mesh(filepath=None):
    if filepath and os.path.exists(filepath):
        return trimesh.load(filepath)
    else:
        return None

# Ho aggiunto il parametro preview=False di default
def render_to_image(mesh, output_file, preview=False):
    # 1. Flat Shading
    try:
        mesh.unmerge_vertices()
    except:
        pass 
    
    # 2. Colore
    mesh.visual.face_colors = OBJECT_COLOR
    
    # 3. Scena
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.set_camera(angles=[np.pi/4, np.pi/4, 0], distance=mesh.scale * 2.5)
    
    # --- LOGICA PREVIEW ---
    # Se preview è True, apre la finestra interattiva PRIMA di salvare
    if preview:
        print("Apertura finestra preview (chiudila per procedere al salvataggio)...")
        scene.show()

    print("Rendering in background...")
    
    try:
        # --- MODIFICA FONDAMENTALE ---
        # Impostando visible=False, trimesh renderizza off-screen senza aprire finestre
        png_data = scene.save_image(resolution=RESOLUTION, visible=False)
    except Exception as e:
        print(f"❌ Errore rendering: {e}")
        return

    # 4. Sfondo Bianco
    print("Applicazione sfondo bianco...")
    try:
        if png_data is None:
            print("❌ Errore: Dati immagine vuoti (possibile problema driver/headless).")
            return

        image_transparent = Image.open(io.BytesIO(png_data))
        final_image = Image.new("RGB", image_transparent.size, "WHITE")
        final_image.paste(image_transparent, (0, 0), image_transparent)
        final_image.save(output_file, "JPEG")
        print(f"✓ Fatto! Immagine salvata in: {output_file}")
    except Exception as e:
        print(f"❌ Errore salvataggio JPG: {e}")