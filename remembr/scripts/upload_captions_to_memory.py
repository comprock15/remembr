import json
import argparse
from memory.memory import MemoryItem
from memory.milvus_memory import MilvusMemory
from memory.text_memory import TextMemory
from memory.video_memory import VideoMemory, ImageMemoryItem
from PIL import Image as PILImage
import numpy as np

def fill_memory_from_json(json_file, collection_name="my_memory", db_ip='127.0.0.1'):
    """
    Заполняет память ReMEmbR данными из JSON файла
    """
    # Инициализация памяти
    memory = MilvusMemory(collection_name, db_ip=db_ip)
    
    # Очистка существующей коллекции
    print(f"Очистка коллекции {collection_name}...")
    memory.reset()
    
    # Загрузка данных из JSON
    print(f"Загрузка данных из {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Добавление элементов в память
    print("Добавление элементов в память...")
    for i, item_data in enumerate(data):
        # Создание MemoryItem
        memory_item = MemoryItem(
            #image=PILImage.fromarray(np.array(item_data['cam0']).astype('uint8'), 'RGB'),
            #image=item_data['cam0'],
            caption=item_data.get("caption", ""),
            time=item_data.get("time", i * 1.0),  # если время не указано, используем последовательное
            position=item_data.get("position", [10.0, 10.0, 10.0]),
            theta=item_data.get("theta", 3.14),
        )
        
        # Вставка в память
        memory.insert(memory_item)
        
        if (i + 1) % 10 == 0:
            print(f"Добавлено {i + 1} элементов...")
    
    print(f"Успешно добавлено {len(data)} элементов в коллекцию {collection_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate ReMEmbR memory from JSON")
    parser.add_argument("--captions", type=str, required=True, help="Path to JSON captions file")
    parser.add_argument("--collection", type=str, default="default_memory", help="Collection name")
    parser.add_argument("--db_ip", type=str, default="127.0.0.1", help="Milvus DB IP")
    
    args = parser.parse_args()
    
    fill_memory_from_json(args.captions, args.collection, args.db_ip)