import json
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

def convert_xml_to_json_universal(xml_content: str) -> Dict[str, Any]:
    root = ET.fromstring(xml_content)

    # Имя файла изображения
    filename = root.findtext('filename', default='')

    fields = []
    for obj in root.findall('object'):
        field_type = obj.findtext('name', default='')
        # Текста в XML нет, оставляем пустым
        text = ''

        bndbox = obj.find('bndbox')
        if bndbox is not None:
            x1 = int(bndbox.findtext('xmin', default='0'))
            y1 = int(bndbox.findtext('ymin', default='0'))
            x2 = int(bndbox.findtext('xmax', default='0'))
            y2 = int(bndbox.findtext('ymax', default='0'))

            field = {
                "field_type": field_type,
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
            fields.append(field)

    return {"image_name": filename, "fields": fields}

def convert_xml_files_in_directory(directory_path: str = ".") -> List[Dict[str, Any]]:
    """
    Конвертирует все XML файлы в указанной папке в JSON формат
    
    Args:
        directory_path: Путь к папке с XML файлами
        
    Returns:
        Список словарей с данными в формате JSON
    """
    results = []
    
    # Проверяем существование папки
    if not os.path.exists(directory_path):
        print(f"Папка {directory_path} не найдена")
        return results
    
    # Ищем все XML файлы в папке
    xml_files = [f for f in os.listdir(directory_path) 
                 if f.lower().endswith('.xml') and os.path.isfile(os.path.join(directory_path, f))]
    
    if not xml_files:
        print(f"В папке {directory_path} не найдено XML файлов")
        return results
    
    print(f"Найдено {len(xml_files)} XML файлов для обработки")
    
    # Обрабатываем каждый XML файл
    for xml_file in xml_files:
        try:
            file_path = os.path.join(directory_path, xml_file)
            
            # Читаем XML файл
            with open(file_path, 'r', encoding='utf-8') as file:
                xml_content = file.read()
            
            # Конвертируем в JSON
            json_result = convert_xml_to_json_universal(xml_content)
            results.append(json_result)
            
            print(f"✓ Обработан файл: {xml_file}")
            
        except Exception as e:
            print(f"✗ Ошибка при обработке файла {xml_file}: {str(e)}")
    
    return results


def save_results_to_file(data: List[Dict[str, Any]], output_file: str = 'data/converted_annotations.json'):
    """
    Сохраняет результаты конвертации в JSON файл
    
    Args:
        data: Данные для сохранения
        output_file: Имя выходного файла
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"✓ Результат сохранен в файл '{output_file}'")
    except Exception as e:
        print(f"✗ Ошибка при сохранении файла: {e}")

if __name__ == '__main__':
    data = convert_xml_files_in_directory('dataset')
    save_results_to_file(data)
