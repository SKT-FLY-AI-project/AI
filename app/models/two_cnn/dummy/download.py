from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from urllib import request
import os
import ssl
import json
import re

ARTS_LIST = 'arts-select.list'
NUMBER_TO_DOWNLOAD = 328
LABELS_JSON = "labels_with_image_paths.json"

# Initialize the JSON file with an empty list if it doesn't exist
if not os.path.exists(LABELS_JSON):
    with open(LABELS_JSON, "w", encoding="utf-8") as json_file:
        json.dump([], json_file)

def download_image_selenium(object_number, title, department, culture, period, object_date, webpage, artist_display_name):
    try:
        # ChromeDriver 설정
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Headless 모드 (브라우저 창 없이 실행)
        chrome_options.add_argument("--no-sandbox")
        
        # ChromeDriver 경로 설정
        service = Service(r"C:\chromedriver-win64\chromedriver.exe")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(webpage)

        # <img> 태그에서 id='artwork__image'인 태그의 src 속성 가져오기
        image_tag = driver.find_element(By.ID, "artwork__image")
        image_url = image_tag.get_attribute("src")

        print(f"Found image URL for {object_number}: {image_url}")

        # SSL 검증 비활성화
        context = ssl._create_unverified_context()
        image_response = request.urlopen(image_url, context=context)

        # 디렉토리 생성
        culture_dir = culture.replace(",", "").replace("/", " ")
        download_dir = f"data/met_art/{culture_dir}"
        os.makedirs(download_dir, exist_ok=True)

        # 고유 이미지 파일 이름 생성
        file_name = f"{object_number}"
        # file_name = f"{title}"
        download_path = os.path.join(download_dir, file_name)

        # 이미지 저장
        with open(download_path, 'wb') as image_file:
            image_file.write(image_response.read())
        
        print(f"Downloaded {download_path}")

        # **레이블과 이미지 경로를 JSON 파일에 저장**
        label_data = {
            "object_number": object_number,
            "title": title,
            "department": department,
            "culture": culture,
            "period": period,
            "object_date": object_date,
            "webpage": webpage,
            "image_url": image_url,
            "file_path": download_path,
            "artist_display_name" : artist_display_name
        }

        # Append the new label data to the JSON file
        with open(LABELS_JSON, "r+", encoding="utf-8") as json_file:
            data = json.load(json_file)
            data.append(label_data)
            json_file.seek(0)
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Error downloading from {webpage} for {culture}: {e}")
    
    finally:
        driver.quit()  # 브라우저 닫기


def main():
    with open(ARTS_LIST, "r", encoding="utf-8") as f:
        arts_to_download = [x.strip() for x in f.readlines()]

    download_count = 0

    for item in arts_to_download:
        # 정규 표현식으로 데이터 파싱
        # match = re.match(r"\( u'(.*?)', u'(.*?)', u'(.*?)', u'(.*?)', u'(.*?)', u'(.*?)', u'(.*?)' \)", item)
        # match = re.match(r"\(\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)'\s*\)", item)
        match = re.match(r"\(\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)'\s*\)", item)


        print(match)
        
        if match:
            object_number = match.group(1)
            title = match.group(2)
            department = match.group(3)
            culture = match.group(4)
            period = match.group(5)
            object_date = match.group(6)
            webpage = match.group(7).replace(" )", "").replace("'", "").strip()
            artist_display_name = match.group(8)  # 추가됨

            print(f"Processing {title} ({culture}) by {artist_display_name}: {webpage}")

            # Selenium을 사용해 이미지 다운로드 및 메타데이터 저장
            download_image_selenium(object_number, title, department, culture, period, object_date, webpage, artist_display_name)
            download_count += 1

            if 0 < NUMBER_TO_DOWNLOAD == download_count:
                print("Reached the download limit.")
                break
        
        else:
            print(f"Failed to parse: {item}")

    print("Download completed.")

if __name__ == "__main__":
    main()