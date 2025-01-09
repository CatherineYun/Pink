import json

# 读取JSON文件
with open('parsed-fgvc-aircraft-2013b-gpt-4-vision-preview-wiki-text-pruned-lbl_combined.json', 'r', encoding='utf-8') as json_file:
    aircraft_data = json.load(json_file)

# 读取txt文件
with open('aircraft_files.txt', 'r') as txt_file:
    lines = txt_file.readlines()

# 创建一个字典来存储结果
result = {}

# 遍历txt文件中的每一行
for line in lines:
    file_name, model = line.strip().split()
    for aircraft in aircraft_data:
        if model in aircraft.get('fine', []):
            result[file_name] = aircraft
            break

# 将结果写入新的JSON文件
with open('/data/Katherine/Pink/out/output.json', 'w', encoding='utf-8') as output_file:
    json.dump(result, output_file, ensure_ascii=False, indent=4)

print("JSON文件已生成：output.json")