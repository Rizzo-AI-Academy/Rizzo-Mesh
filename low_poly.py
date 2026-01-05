

# ----------------------

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