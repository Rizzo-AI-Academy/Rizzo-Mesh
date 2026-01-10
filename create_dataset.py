from low_poly import *
from quantize import *
from utils import *
from pathlib import Path
from ai_captiononer import *
from tqdm import tqdm

DATASET_FOLDER = "dataset_objs_full"
OUT_PUT_FOLDER = "quantized_objs"
CAPTIONS_FOLDER = "captions"

def create_quantized_dataset(dataset_folder: str, output_folder: str):
    """
    Stessa funzionalità della versione os.walk, ma con pathlib.
    """
    dataset_path = Path(dataset_folder)
    out_root     = Path(output_folder)

    for obj_file in tqdm(dataset_path.rglob('*.obj')):          # ricerca ricorsiva
        mesh = get_mesh(str(obj_file))                   # carica la mesh
        if mesh:
            quant_name = obj_file.stem + '_quantized.obj'
            out_file   = out_root / quant_name

            # crea eventuali sottocartelle mancanti
            out_file.parent.mkdir(parents=True, exist_ok=True)

            # salva la mesh quantizzata
            try:
                quantize_mesh(mesh, out_path=str(out_file))
            except Exception as e:
                print(f"Error quantizing {obj_file}: {e}")

def create_captions(dataset_folder: str, output_folder: str):
    dataset_path = Path(dataset_folder)
    out_root     = Path(output_folder)
    index = 0
    for obj_file in dataset_path.rglob('*.obj'):
        mesh = get_mesh(str(obj_file))
        if index < 10:
            index += 1
            if mesh:
                quant_name = obj_file.stem + '_image.jpg'
                out_file   = out_root / quant_name
                render_to_image(mesh, out_file)
                caption = ai_captioning(out_file)
                caption_name = obj_file.stem + '_caption.txt'
                with open(f"{output_folder}/{caption_name}", "w", encoding="utf-8") as f:
                    f.write(caption)

def crea_dataset():
    # Create quantized già realizzate
    # create_quantized_dataset(dataset_folder=DATASET_FOLDER, output_folder=OUT_PUT_FOLDER)
    create_captions(dataset_folder=DATASET_FOLDER, output_folder="img_rendered")

crea_dataset()