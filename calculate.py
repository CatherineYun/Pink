import json

def calculate_accuracy(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_count = 0
    total_count = len(data)

    for item in data:
        image_name = item["image_name"]
        output = item["output"]

        if "others" in image_name and "No" in output:
            correct_count += 1
        elif "others" not in image_name and "Yes" in output:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

if __name__ == "__main__":
    json_file = "/lustre/Katherine/Pink/results_GPS_v3.json"  # 指定你的JSON文件路径
    accuracy = calculate_accuracy(json_file)
    print(f"准确率: {accuracy * 100:.2f}%")