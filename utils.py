import trimesh
import numpy as np
import open3d as o3d
from PIL import Image
import io
import os
import time

RESOLUTION = (1920, 1080)
# Convertiamo colore RGB [30, 144, 255] in float 0-1 per Open3D
OBJECT_COLOR_RGB = [30/255, 144/255, 255/255]

def get_mesh(filepath=None):
    if filepath and os.path.exists(filepath):
        return trimesh.load(filepath)
    else:
        return None

def trimesh_to_open3d(mesh):
    """Converte una mesh Trimesh in Open3D TriangleMesh"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    
    # Calcola normali per shading corretto
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(OBJECT_COLOR_RGB)
    return o3d_mesh

def render_to_image(mesh, output_file, preview=False):
    # 1. Cleanup
    try:
        mesh.unmerge_vertices()
    except:
        pass 

    # 2. Stable Pose (Orientamento)
    try:
        poses, probs = mesh.compute_stable_poses(n_samples=5, threshold=0.02)
        if len(poses) > 0:
            idx = np.argmax(probs)
            mesh.apply_transform(poses[idx])
    except:
        pass

    # Centrare la mesh (Cruciale)
    try:
        mesh.apply_translation(-mesh.bounding_box.centroid)
    except:
        pass 

    # 3. Conversione a Open3D
    o3d_mesh = trimesh_to_open3d(mesh)

    # 4. Setup Visualizer (Headless-ish)
    # Su Windows "visible=False" funziona molto meglio di Pyglet
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=RESOLUTION[0], height=RESOLUTION[1], visible=False)
    vis.add_geometry(o3d_mesh)
    
    # Opzioni di rendering (Opt: Light, Background)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # Bianco
    opt.mesh_show_back_face = True
    
    ctr = vis.get_view_control()

    # 5. MULTI-VIEW RENDERING
    # Generiamo 3 viste con angolazioni MIGLIORATE per evitare vista "da sotto":
    # Assumiamo sistema di coordinate tipico Z-up (o adattiamo).
    
    views = []
    
    # 5. MULTI-VIEW RENDERING
    # 3 VISTE ORTOGONALI PURE (Richiesta Utente: Frontale, Laterale, Dall'alto)
    # Assi X, Y, Z.
    
    views = []
    
    view_configs = [
        # 1. FRONTALE (Asse -Y)
        {"front": [0.0, -1.0, 0.0], "up": [0, 0, 1], "zoom": 0.6},
        
        # 2. LATERALE (Asse +X)
        {"front": [1.0, 0.0, 0.0], "up": [0, 0, 1], "zoom": 0.6},
        
        # 3. DALL'ALTO (Asse +Z)
        # Nota: Per la vista dall'alto, l'UP vector deve essere l'asse Y per orientarla correttamente
        {"front": [0.0, 0.0, 1.0], "up": [0, 1, 0], "zoom": 0.6}
    ]
    
    print(f"Rendering 3 Strictly Orthogonal Views (Front/Side/Top) for {output_file.name}...")
    
    for i, cfg in enumerate(view_configs):
        try:
            # Set View
            ctr.set_lookat([0, 0, 0]) 
            ctr.set_front(cfg["front"])
            ctr.set_up(cfg["up"])
            ctr.set_zoom(cfg["zoom"])
            
            # Update
            vis.poll_events()
            vis.update_renderer()
            
            # Capture
            img_data = vis.capture_screen_float_buffer(do_render=True)
            
            img_array = (np.asarray(img_data) * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            if img:
                views.append(img)
            else:
                views.append(Image.new("RGB", RESOLUTION, "WHITE"))
                
        except Exception as e:
            print(f"❌ Errore vista {i+1}: {e}")
            views.append(Image.new("RGB", RESOLUTION, "WHITE"))

    vis.destroy_window()
    print(f"  Closed window for {output_file.name}")

    # 6. STITCHING (Unione Immagini)
    try:
        if not views:
            return

        total_width = views[0].width * len(views)
        max_height = views[0].height
        
        combined_image = Image.new("RGB", (total_width, max_height), "WHITE")
        
        current_x = 0
        for img in views:
            combined_image.paste(img, (current_x, 0))
            current_x += img.width
            
        target_width = 2000
        ratio = target_width / total_width
        target_height = int(max_height * ratio)
        combined_image = combined_image.resize((target_width, target_height), Image.LANCZOS)
        
        combined_image.save(output_file, "JPEG")
        print(f"✓ Salvato Multi-View (O3D): {output_file.name}")
        
    except Exception as e:
        print(f"❌ Errore salvataggio combined: {e}")