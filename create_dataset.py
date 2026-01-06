from low_poly import *
from quantize import *
from utils import *
from pathlib import Path
from ai_captiononer import *


DATASET_FOLDER = "dataset_objs_full"
OUT_PUT_FOLDER = "quantized_objs"
CAPTIONS_FOLDER = "captions"

def create_quantized_dataset(dataset_folder: str, output_folder: str):
    """
    Stessa funzionalità della versione os.walk, ma con pathlib.
    """
    dataset_path = Path(dataset_folder)
    out_root     = Path(output_folder)

    for obj_file in dataset_path.rglob('*.obj'):          # ricerca ricorsiva
        mesh = get_mesh(str(obj_file))                   # carica la mesh

        # calcola il percorso di output mantenendo la struttura
        rel_path = obj_file.relative_to(dataset_path)
        quant_name = rel_path.with_name(rel_path.stem + '_quantized.obj')
        out_file   = out_root / quant_name

        # crea eventuali sottocartelle mancanti
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # salva la mesh quantizzata
        quantize_mesh(mesh, out_path=str(out_file))

def create_captions(dataset_folder: str, output_folder: str):
    dataset_path = Path(dataset_folder)
    out_root     = Path(output_folder)
    for obj_file in dataset_path.rglob('*.obj'):
        mesh = get_mesh(str(obj_file))
        rel_path = obj_file.relative_to(dataset_path)
        quant_name = rel_path.with_name(rel_path.stem + '_image.jpg')
        out_file   = out_root / quant_name
        render_to_image(mesh, out_file)
        caption = ai_captioning(out_file)
        rel_path = obj_file.relative_to(dataset_path)
        caption_name = rel_path.with_name(rel_path.stem + '_caption.txt')
        with open(f"{output_folder}/{caption_name}", "w", encoding="utf-8") as f:
            f.write(caption)
        
        # caption = create_caption(imgs)
        # save caption as file .txt

def crea_dataset():
    create_quantized_dataset(dataset_folder=DATASET_FOLDER, output_folder=OUT_PUT_FOLDER)
    create_captions(dataset_folder=DATASET_FOLDER, output_folder=OUT_PUT_FOLDER)

crea_dataset()