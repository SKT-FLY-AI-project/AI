import os

# 클래스 매핑 정의
class_mapping = {
    "artwork": 0,
}

# 라벨 파일을 수정하는 함수
def update_labels(label_dir):
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            label_path = os.path.join(label_dir, filename)

            with open(label_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                coords = line.strip().split()
                class_name = "artwork"  # 예제: artwork로 분류
                class_id = class_mapping.get(class_name, 0)
                new_lines.append(f"{class_id} " + " ".join(coords))

            # 새로운 라벨 데이터 저장
            with open(label_path, "w") as f:
                f.write("\n".join(new_lines))

# 실행
update_labels("path/to/labels")