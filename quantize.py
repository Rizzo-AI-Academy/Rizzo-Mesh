import numpy as np
import trimesh

def quantize_mesh(
    mesh: trimesh.Trimesh,
    out_path: str,
    bins: int = 256,                 # 256 -> 0..255
    normalize_mode: str = "unit_cube",
    canonicalize_faces: bool = True,
    remove_degenerate: bool = True,
    remove_duplicate_faces: bool = True,
):
    """
    Stile LLaMA-Mesh + robustezza:
    - quantizza vertici a int [0..bins-1] (senza salvare scale/offset per mesh)
    - ordina vertici per z,y,x
    - MERGE vertici duplicati (dopo quantizzazione) preservando l'ordine z,y,x
    - costruisce mappa old->new e rimappa facce
    - (opz.) ruota facce mettendo il minimo indice per primo (senza reverse)
    - ordina facce per chiave (indici crescenti) senza cambiare l'ordine interno
    - salva OBJ con v interi e f
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh deve essere trimesh.Trimesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh vuota (vertici o facce mancanti)")

    # Consiglio: lavora su triangoli per evitare triangolazioni imprevedibili nei viewer
    m = mesh.copy()
    try:
        m = m.triangulate()
    except Exception:
        pass

    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)

    # 1) centra
    v0 = v - v.mean(axis=0)

    # 2) normalizza deterministica (no meta)
    if normalize_mode == "unit_cube":
        extent = v0.max(axis=0) - v0.min(axis=0)
        s = float(np.max(extent)) if np.max(extent) != 0 else 1.0
        vN = v0 * (2.0 / s)               # ~[-1,1]
    elif normalize_mode == "unit_sphere":
        r = float(np.linalg.norm(v0, axis=1).max()) if len(v0) else 1.0
        r = 1.0 if r == 0 else r
        vN = v0 / r
    else:
        raise ValueError("normalize_mode deve essere 'unit_cube' o 'unit_sphere'")

    vN = np.clip(vN, -1.0, 1.0)

    # 3) quantizza -> int
    q = np.rint((vN + 1.0) * 0.5 * (bins - 1)).astype(np.int32)
    q = np.clip(q, 0, bins - 1).astype(np.uint8)

    # 4) ordina vertici per z,y,x
    order = np.lexsort((q[:, 0], q[:, 1], q[:, 2]))  # z primario
    q_sorted_full = q[order]

    # mappa: old_index -> sorted_position
    old_to_sortedpos = np.empty(len(order), dtype=np.int64)
    old_to_sortedpos[order] = np.arange(len(order), dtype=np.int64)

    # 5) MERGE duplicati dopo quantizzazione, preservando ordine z,y,x
    q_unique, sortedpos_to_unique = _stable_unique_rows_preserve_order(q_sorted_full)

    # mappa finale: old_vertex_index -> unique_vertex_index
    old_to_unique = sortedpos_to_unique[old_to_sortedpos]

    # 6) rimappa facce su indici dei vertici unici
    f_new = old_to_unique[f]

    # 7) opzionale: elimina facce degenerate (indici ripetuti)
    if remove_degenerate:
        # per triangoli: degenerate se almeno due indici uguali
        deg = (f_new[:, 0] == f_new[:, 1]) | (f_new[:, 0] == f_new[:, 2]) | (f_new[:, 1] == f_new[:, 2])
        f_new = f_new[~deg]

    # 8) canonicalizza facce (min indice per primo) SENZA invertire
    if canonicalize_faces:
        f_canon = np.vstack([_rotate_face_min_first(face) for face in f_new])
    else:
        f_canon = f_new

    # 9) ordina facce per chiave: (min, second, third, ...)
    #    chiave calcolata su indici ordinati, ma output resta f_canon
    face_keys = np.sort(f_canon, axis=1)
    face_order = np.lexsort(face_keys.T[::-1])
    f_sorted = f_canon[face_order]

    # 10) rimuovi facce duplicate (stessa tripla di indici, indipendente dal ciclo)
    if remove_duplicate_faces and len(f_sorted) > 0:
        # usa la chiave ordinata per deduplicare
        keys = np.sort(f_sorted, axis=1)
        _, uniq_idx = np.unique(keys, axis=0, return_index=True)
        f_sorted = f_sorted[np.sort(uniq_idx)]

    # 11) salva OBJ (VERTICI INT + FACCE)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("# Quantized OBJ int, vertices sorted z-y-x, merged duplicates, faces remapped+sorted\n")
        fp.write(f"# bins: {bins}\n")
        fp.write(f"# normalize_mode: {normalize_mode}\n")
        fp.write(f"# canonicalize_faces: {canonicalize_faces}\n")

        for x, y, z in q_unique:
            fp.write(f"v {int(x)} {int(y)} {int(z)}\n")

        for face in f_sorted:
            fp.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")

    return {
        "out_path": out_path,
        "bins": bins,
        "normalize_mode": normalize_mode,
        "vertices_out": int(len(q_unique)),
        "faces_out": int(len(f_sorted)),
    }

def load_quantized_obj_int255(path: str):
    """
    Carica un OBJ con vertici interi 0..255 (righe 'v i j k')
    e facce 'f ...' (solo indici vertex).
    Ritorna: (q_vertices_uint8, faces_int64, normalize_mode_str_or_None)
    """
    qv = []
    faces = []
    normalize_mode = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # se presente, prende la info scritta dalla funzione di export
                if line.lower().startswith("# normalize_mode:"):
                    normalize_mode = line.split(":", 1)[1].strip()
                continue

            if line.startswith("v "):
                parts = line.split()
                # supporta solo 3 valori
                x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                qv.append((x, y, z))

            elif line.startswith("f "):
                # supporto: "f 1 2 3" oppure "f 1/.. 2/.. 3/.."
                parts = line.split()[1:]
                idx = []
                for p in parts:
                    # prende solo la parte prima di eventuali / (vt/vn)
                    i = int(p.split("/")[0]) - 1  # OBJ è 1-based
                    idx.append(i)
                faces.append(idx)

    qv = np.array(qv, dtype=np.uint8)
    faces = np.array(faces, dtype=np.int64)
    return qv, faces, normalize_mode


def visualize_quantized_obj_int255(path: str, show: bool = True):
    """
    Visualizza un OBJ quantizzato (v in 0..255) interpretandolo in [-1,1].
    """
    qv, faces, normalize_mode = load_quantized_obj_int255(path)

    # De-quantizzazione in spazio normalizzato [-1,1]
    v = (qv.astype(np.float64) / 255.0) * 2.0 - 1.0

    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)

    # print(f"Loaded quantized OBJ: {path}")
    # print(f"normalize_mode (comment): {normalize_mode}")
    # print(f"verts: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

    if show:
        mesh.show()

    return mesh

def _rotate_face_min_first(face: np.ndarray) -> np.ndarray:
    """
    Ruota ciclicamente la faccia per mettere il minimo indice in prima posizione,
    preservando l'ordine (NON fa reverse).
    """
    face = np.asarray(face, dtype=np.int64)
    k = int(np.argmin(face))
    return np.roll(face, -k)

def _stable_unique_rows_preserve_order(a: np.ndarray):
    """
    Unique righe 2D preservando l'ordine già presente in 'a'.
    Ritorna: (unique_rows, inverse) dove inverse mappa ogni riga di 'a' -> indice in unique_rows
    """
    if len(a) == 0:
        return a, np.zeros((0,), dtype=np.int64)

    # a è già ordinato z,y,x: i duplicati saranno adiacenti
    diff = np.any(a[1:] != a[:-1], axis=1)
    first_mask = np.concatenate(([True], diff))
    unique_rows = a[first_mask]

    # inverse: per ogni riga, a quale blocco (unique) appartiene
    inverse = np.cumsum(first_mask) - 1
    return unique_rows, inverse.astype(np.int64)
