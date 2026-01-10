import random
from low_poly import *
from quantize import *
from utils import *
from pathlib import Path
from ai_captiononer import *
from tqdm import tqdm
import asyncio
import json


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

# --- NUOVA ARCHITETTURA: SEQUENZIALE (Render) -> PARALLELA (AI) ---

import multiprocessing

def _render_wrapper(obj_file_str, out_file_str):
    """Wrapper per eseguire il render in un processo separato"""
    try:
        # Ricarichiamo le librerie necessarie nel processo figlio se serve, 
        # ma su Windows con 'spawn' (default) dovrebbe andare bene se importate globalmente.
        # Meglio essere sicuri di usare i path stringa
        mesh = get_mesh(obj_file_str)
        if mesh:
            out_path = Path(out_file_str)
            render_to_image(mesh, out_path)
    except Exception as e:
        print(f"Error in render process: {e}")

def render_all_sequential(files: list, out_root: Path):
    """
    Esegue il rendering usando PROCESSI SEPARATI con TIMEOUT.
    Questo è l'unico modo per uccidere un render Open3D bloccato.
    """
    print(f"🎨 Avvio Rendering (Process Isolation Mode) di {len(files)} oggetti...")
    generated_images = []
    
    for obj_file in tqdm(files, desc="Rendering Safe"):
        # Path output
        quant_name = obj_file.stem + '_image.jpg'
        out_file = out_root / quant_name
        
        # Skip esistenti
        if out_file.exists():
            # print(f"⏩ Skip: {quant_name}")
            generated_images.append((obj_file, out_file))
            continue
            
        print(f"🎥 Rendering: {obj_file.name}")
        
        # Creiamo un processo per questo singolo render
        # Passiamo stringhe per evitare problemi di pickling con Path complessi
        p = multiprocessing.Process(target=_render_wrapper, args=(str(obj_file), str(out_file)))
        p.start()
        
        # Attendiamo max 15 secondi
        p.join(timeout=15)
        
        if p.is_alive():
            print(f"💀 TIMEOUT! Uccisione processo bloccato per {obj_file.name}")
            p.terminate()
            p.join() # cleanup
        else:
            # Se è finito e il file esiste, successo
            if out_file.exists():
                generated_images.append((obj_file, out_file))
            else:
                print(f"⚠ Processo finito ma file non creato: {obj_file.name}")
                
    return generated_images

async def caption_worker(queue, out_root, pbar, worker_id):
    """Worker solo per AI, safe da parallelizzare"""
    # print(f"[AI-Worker {worker_id}] Ready.")
    while True:
        try:
            # Attendiamo il task. Se veniamo cancellati qui, 
            # solleva CancelledError e usciamo puliti.
            item = await queue.get()
        except asyncio.CancelledError:
            # Il main ci ha detto di smettere
            # print(f"[AI-Worker {worker_id}] Stopping...")
            break
            
        try:
            # spacchettiamo l'item
            obj_file, img_path = item
            
            # Captioning
            abs_path = str(img_path.resolve())
            # print(f"[AI-Worker {worker_id}] Captioning {img_path.name}...")
            
            caption = await ai_captioning(abs_path)
            
            if not caption:
                caption = "Nessuna descrizione generata."
            
            # Salvataggio
            caption_name = obj_file.stem + '_caption.txt'
            out_caption_path = out_root / caption_name
            
            with open(out_caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
                
            print(f"✓ [AI-Worker {worker_id}] Done: {caption_name}")
            
        except Exception as e:
            print(f"⚠ [AI-Worker {worker_id}] Error: {e}")
        finally:
            # Segnaliamo che QUESTO task specifico è finito
            pbar.update(1)
            queue.task_done()

async def caption_all_parallel(generated_images, out_root):
    """
    Processa le immagini generate in parallelo con l'AI.
    """
    print(f"🤖 Avvio AI Captioning Parallelo su {len(generated_images)} immagini...")
    
    queue = asyncio.Queue()
    for item in generated_images:
        queue.put_nowait(item)
        
    # Qui possiamo osare di più con i worker perché sono solo chiamate HTTP
    num_ai_workers = 10 
    pbar = tqdm(total=len(generated_images), desc="AI Processing")
    
    workers = []
    for i in range(num_ai_workers):
        task = asyncio.create_task(caption_worker(queue, out_root, pbar, i))
        workers.append(task)
        
    await queue.join()
    
    for t in workers: t.cancel()
    pbar.close()

async def create_captions_main(dataset_folder: str, output_folder: str, num_examples=20):
    dataset_path = Path(dataset_folder)
    out_root = Path(output_folder)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Trova i file
    if num_examples == -1:
         files = list(dataset_path.rglob('*.obj'))
    else:
         files = list(dataset_path.rglob('*.obj'))[:num_examples]
    
    # 2. FASE RENDER (Sequenziale - Main Thread)
    # È bloccante, ma evita i crash OpenGL
    images_ready = render_all_sequential(files, out_root)
    
    # 3. FASE AI (Parallela - AsyncIO)
    if images_ready:
        await caption_all_parallel(images_ready, out_root)
    else:
        print("Nessuna immagine generata da processare.")

def create_captions(dataset_folder: str, output_folder: str):
    asyncio.run(create_captions_main(dataset_folder, output_folder, num_examples=20))

def final_dataset_for_llm():
    # read all .txt files inside the folder img_rendered
    captions_path = Path("img_rendered")
    dataset = []
    for txt_file in captions_path.rglob('*.txt'):
        obj_id = txt_file.name.split("_")[0]

        with open(f"quantized_objs/{obj_id}_quantized.obj", 'r', encoding='utf-8') as f:
            # read file from row 5
            obj_content = "".join(f.readlines()[4:])
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            caption_content = f.read().strip()

        start_sentences = ["Crea un modello 3D di ", "Crea un modello 3D che rappresenta: "]
        context_sencence = ["Cos'è quest'oggetto:", "Descrivi quest' oggetto:"]

        # get a random start sencente
        start_sentence = random.choice(start_sentences)
        context_sentence = random.choice(context_sencence)

        question1 = f"{start_sentence}{caption_content}"
        response1 = f"\nEcco il tuo oggetto 3D:\n{obj_content}"
        question2 = f"{context_sentence}\n{obj_content}"
        response2 = f"\nRappresenta: {caption_content}"
        dataset.append([{"role":"user", "content": question1,}, {"role": "assistant","content": response1}])
        dataset.append([{"role":"user", "content": question2,}, {"role": "assistant","content": response2}])
    # save the database into a json file locally
    
    with open("database.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        

            

        

def crea_dataset():
    # Create quantized già realizzate
    # create_quantized_dataset(dataset_folder=DATASET_FOLDER, output_folder=OUT_PUT_FOLDER)
    create_captions(dataset_folder=DATASET_FOLDER, output_folder="img_rendered")
    final_dataset_for_llm()

if __name__ == "__main__":
    crea_dataset()