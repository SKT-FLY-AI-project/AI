# $ python models/two_cnn/data/cnn_split_data.py
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np
from sklearn.model_selection import train_test_split
import re

def sanitize_title(title, max_length=50):
    title = re.sub(r'[^a-zA-Z0-9가-힣_]', '_', title)  # 특수문자를 _로 대체
    return title[:max_length]  # 제목을 최대 max_length로 제한

# JSON 파일 경로
json_file_path = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\labels_with_image_paths.json"

# train/validation 데이터 폴더 경로
train_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_train_data"
val_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_validation_data"

# 폴더 생성 함수
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 폴더 생성
create_dir(train_dir)
create_dir(val_dir)

# JSON 데이터 로드
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 이미지 증강 설정
datagen = ImageDataGenerator(
    rotation_range=10,         # 이미지를 0~10도 사이에서 무작위로 회전합니다.
    width_shift_range=0.05,    # 이미지의 가로 방향으로 최대 5%만 이동합니다.
    height_shift_range=0.05,   # 이미지의 세로 방향으로 최대 5%만 이동합니다.
    shear_range=0.0,           # 왜곡 변형은 제거합니다 (작품의 형태가 유지되도록).
    zoom_range=0.1,            # 확대/축소 범위를 90%~110%로 제한합니다.
    horizontal_flip=True,      # 수평 반전은 유지합니다.
    fill_mode='nearest'        # 빈 공간은 가장 가까운 픽셀 값으로 채웁니다.
)

# 증강할 이미지 개수 설정
num_augmented_images = 10  # 각 이미지당 10개의 증강 이미지 생성

# JSON 파일이 있는 디렉터리를 기준으로 file_path를 절대 경로로 변환
base_dir = os.path.dirname(json_file_path)

# 이미지 증강 및 저장 함수
def augment_and_save_images(data, train_dir, val_dir):
    for item in data:
        # 파일 경로 및 클래스 이름 설정
        title = sanitize_title(item["title"])
        relative_file_path = item["file_path"]
        file_path = os.path.abspath(os.path.join(base_dir, relative_file_path))  # 절대 경로로 변환

        if not os.path.exists(file_path):
            print(f"이미지 파일을 찾을 수 없습니다: {file_path}")
            continue

        # 이미지 로드 및 크기 조정
        img = load_img(file_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 150, 150, 3)

        # 저장할 폴더 생성
        train_class_dir = os.path.join(train_dir, title)
        val_class_dir = os.path.join(val_dir, title)
        create_dir(train_class_dir)
        create_dir(val_class_dir)

        # **원래 이미지**를 train 데이터에 저장
        original_img = array_to_img(img_array[0])
        original_img.save(os.path.join(train_class_dir, f"{title}_original.jpg"))
        print(f"{title} - 원래 이미지 저장 완료!")

        # 증강 이미지 생성
        augmented_images = []
        for batch in datagen.flow(img_array, batch_size=1):
            augmented_images.append(batch[0])  # 증강된 이미지를 리스트에 추가
            if len(augmented_images) >= num_augmented_images:
                break

        # 증강된 이미지를 train/validation으로 나누기
        train_images, val_images = train_test_split(augmented_images, test_size=0.2, random_state=42)

        # train 데이터 저장
        for i, img_array in enumerate(train_images):
            img = array_to_img(img_array)
            img.save(os.path.join(train_class_dir, f"{title}_train_{i + 1}.jpg"))

        # validation 데이터 저장
        for i, img_array in enumerate(val_images):
            img = array_to_img(img_array)
            img.save(os.path.join(val_class_dir, f"{title}_val_{i + 1}.jpg"))

        print(f"{title} - 증강 및 저장 완료!")

# 증강 및 저장 실행
augment_and_save_images(data, train_dir, val_dir)
print("모든 이미지 증강 및 데이터셋 정리가 완료되었습니다!")

# import os
# import json
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# import numpy as np
# from sklearn.model_selection import train_test_split
# import re

# def sanitize_title(title, max_length=50):
#     title = re.sub(r'[^a-zA-Z0-9가-힣_]', '_', title)  # 특수문자를 _로 대체
#     return title[:max_length]  # 제목을 최대 max_length로 제한

# # JSON 파일 경로
# json_file_path = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\labels_with_image_paths.json"

# # train/validation 데이터 폴더 경로
# train_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_train_data"
# val_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_validation_data"

# # 폴더 생성 함수
# def create_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # 폴더 생성
# create_dir(train_dir)
# create_dir(val_dir)

# # JSON 데이터 로드
# with open(json_file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 이미지 증강 설정
# datagen = ImageDataGenerator(
#     rotation_range=10,         # 이미지를 0~10도 사이에서 무작위로 회전합니다.
#     width_shift_range=0.05,    # 이미지의 가로 방향으로 최대 5%만 이동합니다.
#     height_shift_range=0.05,   # 이미지의 세로 방향으로 최대 5%만 이동합니다.
#     shear_range=0.0,           # 왜곡 변형은 제거합니다 (작품의 형태가 유지되도록).
#     zoom_range=0.1,            # 확대/축소 범위를 90%~110%로 제한합니다.
#     horizontal_flip=True,      # 수평 반전은 유지합니다.
#     fill_mode='nearest'        # 빈 공간은 가장 가까운 픽셀 값으로 채웁니다.
# )

# # 증강할 이미지 개수 설정
# num_augmented_images = 10  # 각 이미지당 10개의 증강 이미지 생성

# # JSON 파일이 있는 디렉터리를 기준으로 file_path를 절대 경로로 변환
# base_dir = os.path.dirname(json_file_path)

# # 이미지 증강 및 저장 함수
# def augment_and_save_images(data, train_dir, val_dir):
#     for item in data:
#         # 파일 경로 및 클래스 이름 설정
#         # title = item["title"].replace("|", "_").replace(" ", "_").replace(":", "_")
#         title = sanitize_title(item["title"])
#         relative_file_path = item["file_path"]
#         file_path = os.path.abspath(os.path.join(base_dir, relative_file_path))  # 절대 경로로 변환

#         if not os.path.exists(file_path):
#             print(f"이미지 파일을 찾을 수 없습니다: {file_path}")
#             continue

#         # 이미지 로드 및 크기 조정
#         img = load_img(file_path, target_size=(150, 150))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 150, 150, 3)

#         # 증강 이미지 생성
#         augmented_images = []
#         for batch in datagen.flow(img_array, batch_size=1):
#             augmented_images.append(batch[0])  # 증강된 이미지를 리스트에 추가
#             if len(augmented_images) >= num_augmented_images:
#                 break


#         # 증강된 이미지를 train/validation으로 나누기
#         train_images, val_images = train_test_split(augmented_images, test_size=0.2, random_state=42)

#         # 저장할 폴더 생성
#         train_class_dir = os.path.join(train_dir, title)
#         val_class_dir = os.path.join(val_dir, title)
#         create_dir(train_class_dir)
#         create_dir(val_class_dir)

#         # train 데이터 저장
#         for i, img_array in enumerate(train_images):
#             img = array_to_img(img_array)
#             img.save(os.path.join(train_class_dir, f"{title}_train_{i + 1}.jpg"))

#         # validation 데이터 저장
#         for i, img_array in enumerate(val_images):
#             img = array_to_img(img_array)
#             img.save(os.path.join(val_class_dir, f"{title}_val_{i + 1}.jpg"))

#         print(f"{title} - 증강 및 저장 완료!")

# # 증강 및 저장 실행
# augment_and_save_images(data, train_dir, val_dir)
# print("모든 이미지 증강 및 데이터셋 정리가 완료되었습니다!")

# import os
# import json
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# import numpy as np
# from sklearn.model_selection import train_test_split

# # JSON 파일 경로
# json_file_path = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\labels_with_image_paths.json"

# # train/validation 데이터 폴더 경로
# train_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_train_data"
# val_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_validation_data"

# # 폴더 생성 함수
# def create_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # 폴더 생성
# create_dir(train_dir)
# create_dir(val_dir)

# # JSON 데이터 로드
# with open(json_file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 이미지 증강 설정
# datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # 증강할 이미지 개수 설정
# num_augmented_images = 10  # 각 이미지당 10개의 증강 이미지 생성

# # 이미지 증강 및 분할 함수
# def augment_and_save_images(data, train_dir, val_dir):
#     for item in data:
#         title = item["title"].replace("|", "_").replace(" ", "_").replace(":", "_")  # 폴더 이름에 사용할 수 없는 문자 대체
#         file_path = item["file_path"]
        
#         if not os.path.exists(file_path):
#             print(f"이미지 파일을 찾을 수 없습니다: {file_path}")
#             continue
        
#         # 이미지 로드
#         img = load_img(file_path, target_size=(150, 150))  # 크기 조정
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 (1, 150, 150, 3)
        
#         # 증강 이미지 생성
#         augmented_images = []
#         for batch in datagen.flow(img_array, batch_size=1):
#             augmented_images.append(batch[0])  # 증강된 이미지를 리스트에 추가
#             if len(augmented_images) >= num_augmented_images:
#                 break

#         # 증강된 이미지를 train/validation으로 나누기
#         train_images, val_images = train_test_split(augmented_images, test_size=0.2, random_state=42)
        
#         # 저장할 폴더 생성
#         train_class_dir = os.path.join(train_dir, title)
#         val_class_dir = os.path.join(val_dir, title)
#         create_dir(train_class_dir)
#         create_dir(val_class_dir)

#         # 이미지 저장
#         for i, img_array in enumerate(train_images):
#             img = array_to_img(img_array)
#             img.save(os.path.join(train_class_dir, f"{title}_train_{i + 1}.jpg"))
        
#         for i, img_array in enumerate(val_images):
#             img = array_to_img(img_array)
#             img.save(os.path.join(val_class_dir, f"{title}_val_{i + 1}.jpg"))
        
#         print(f"{title} - 증강 및 저장 완료!")

# # 증강 및 저장 실행
# augment_and_save_images(data, train_dir, val_dir)
# print("모든 이미지 증강 및 데이터셋 정리가 완료되었습니다!")
