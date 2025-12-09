import trimesh
import numpy as np
import os
import io
from PIL import Image

# --- CONFIGURAZIONE ---
FILE_PATH = r"files\eevee_lowpoly_flowalistik.STL"
OUTPUT_IMG = "render_contrast.jpg"

# METODO 1: Quadric Decimation (raccomandato)
# Percentuale di RIDUZIONE: 0.5 = togli 50% delle facce
# 0.1 = togli 10%, 0.9 = togli 90% (molto low-poly)
REDUCTION_PERCENT = 1  # Riduci del 50%

# METODO 2: Voxel-based (alternativo)
# Più alto = più low-poly (prova 0.01 - 0.1)
VOXEL_SIZE = 0.1
# Scegli il metodo: 'quadric' o 'voxel' o 'none' (per non semplificare)
METODO = 'quadric'

RESOLUTION = (1920, 1080)
OBJECT_COLOR = [30, 144, 255, 255]  # Blu
# ----------------------

def get_mesh(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"Caricamento di {filepath}...")
        return trimesh.load(filepath)
    else:
        print("Nessun file trovato. Scarico esempio...")
        return trimesh.load_remote('https://github.com/mikedh/trimesh/raw/master/models/featuretype.STL')

def make_low_poly_quadric(mesh, reduction):
    """
    Metodo 1: Quadric Edge Collapse Decimation
    Più preciso e mantiene meglio la forma originale
    """
    original_faces = len(mesh.faces)
    target_faces = int(original_faces * (1 - reduction))
    
    print(f"Semplificazione Quadric: {original_faces} -> ~{target_faces} facce (riduzione {reduction*100}%)...")
    
    # Validazione
    if reduction <= 0 or reduction >= 1:
        print(f"⚠ REDUCTION_PERCENT deve essere tra 0.01 e 0.99. Valore attuale: {reduction}")
        return mesh
    
    try:
        # CORRETTO: usa face_count (numero assoluto) NON target_reduction
        mesh_simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        
        if len(mesh_simplified.faces) == 0:
            print("Errore: mesh vuota dopo semplificazione. Uso originale.")
            return mesh
            
        print(f"✓ Semplificazione riuscita: {original_faces} -> {len(mesh_simplified.faces)} facce.")
        return mesh_simplified
        
    except AttributeError:
        print("❌ Errore: simplify_quadric_decimation non disponibile.")
        print("Installa con: pip install pyfqmr")
        print("Uso mesh originale.")
        return mesh
    except Exception as e:
        print(f"❌ Errore Quadric Decimation: {e}")
        print("Uso mesh originale.")
        return mesh

def make_low_poly_voxel(mesh, voxel_size):
    """
    Metodo 2: Voxelizzazione (più robusto ma meno preciso)
    Converte in voxel e poi ricostruisce la mesh
    """
    print(f"Semplificazione via Voxel (dimensione: {voxel_size})...")
    
    try:
        # Calcola la dimensione del voxel basata sulla scala dell'oggetto
        pitch = mesh.scale * voxel_size
        
        # Voxelizza la mesh
        voxelized = mesh.voxelized(pitch=pitch)
        
        # Riconverti in mesh
        mesh_simplified = voxelized.marching_cubes
        
        if len(mesh_simplified.faces) == 0:
            print("Errore: voxel size troppo grande. Uso originale.")
            return mesh
            
        print(f"✓ Voxelizzazione riuscita: {len(mesh.faces)} -> {len(mesh_simplified.faces)} facce.")
        return mesh_simplified
        
    except Exception as e:
        print(f"❌ Errore Voxelizzazione: {e}. Uso mesh originale.")
        return mesh

def make_low_poly_simple(mesh, max_distance=None):
    """
    Metodo 3: Merge Vertices (fallback semplice)
    Unisce i vertici vicini - meno efficace ma sempre funziona
    """
    print(f"Semplificazione via Merge Vertices...")
    
    try:
        # Crea una copia per non modificare l'originale
        mesh_copy = mesh.copy()
        
        # Unisci i vertici vicini
        mesh_copy.merge_vertices()
        
        # Rimuovi facce duplicate e degenerate
        mesh_copy.remove_duplicate_faces()
        mesh_copy.remove_degenerate_faces()
        
        # Rimuovi vertici non referenziati
        mesh_copy.remove_unreferenced_vertices()
        
        print(f"✓ Merge riuscito: {len(mesh.faces)} -> {len(mesh_copy.faces)} facce.")
        return mesh_copy
        
    except Exception as e:
        print(f"❌ Errore Merge: {e}. Uso mesh originale.")
        return mesh

def render_scene_white_bg(mesh, output_file):
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

def main():
    mesh = get_mesh(FILE_PATH)
    
    # Gestione Scene multi-oggetto
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        else:
            print("❌ Errore: Scene vuota")
            return

    print(f"\n📊 Mesh originale: {len(mesh.vertices)} vertici, {len(mesh.faces)} facce")
    print(f"📏 Dimensioni: {mesh.bounds[0]} -> {mesh.bounds[1]}")
    print(f"🔧 Metodo selezionato: {METODO.upper()}\n")
    
    # Scegli il metodo di semplificazione
    if METODO == 'quadric':
        mesh_low = make_low_poly_quadric(mesh, REDUCTION_PERCENT)
    elif METODO == 'voxel':
        mesh_low = make_low_poly_voxel(mesh, VOXEL_SIZE)
    elif METODO == 'none':
        print("Nessuna semplificazione richiesta.")
        mesh_low = mesh
    else:
        # Fallback: metodo semplice
        mesh_low = make_low_poly_simple(mesh)
    
    # Salva la mesh semplificata (opzionale)
    # output_stl = FILE_PATH.replace('.STL', '_lowpoly.STL')
    # mesh_low.export(output_stl)
    # print(f"✓ Mesh salvata: {output_stl}")
    
    render_scene_white_bg(mesh_low, OUTPUT_IMG)

if __name__ == "__main__":
    main()