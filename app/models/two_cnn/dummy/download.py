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

def download_image_selenium(object_number, title, department, culture, period, object_date, webpage):
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
        file_name = f"{object_number}_{image_url.split('/')[-1]}"
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
            "file_path": download_path  # 이미지의 로컬 경로
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
        match = re.match(r"\(\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)',\s*u'(.*?)'\s*\)", item)


        print(match)
        
        if match:
            object_number = match.group(1)
            title = match.group(2)
            department = match.group(3)
            culture = match.group(4)
            period = match.group(5)
            object_date = match.group(6)
            webpage = match.group(7).replace(" )", "").replace("'", "").strip()
            print(f"Processing {title} ({culture}): {webpage}")

            # Selenium을 사용해 이미지 다운로드 및 메타데이터 저장
            download_image_selenium(object_number, title, department, culture, period, object_date, webpage)
            download_count += 1

            if 0 < NUMBER_TO_DOWNLOAD == download_count:
                print("Reached the download limit.")
                break
        
        else:
            print(f"Failed to parse: {item}")

    print("Download completed.")

if __name__ == "__main__":
    main()





"""
이미지 덮어쓰기 이슈 발생,,
이미지만 저장되는 이슈 발생,,

근데 돌아는 감!
"""

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from urllib import request
# import os
# import ssl
# import csv

# ARTS_LIST = 'arts-select.list'
# NUMBER_TO_DOWNLOAD = 100

# def download_image_selenium(culture, webpage):
#     try:
#         # ChromeDriver 설정
#         chrome_options = Options()
#         chrome_options.add_argument("--headless")  # Headless 모드 (브라우저 창 없이 실행)
#         chrome_options.add_argument("--no-sandbox")
        
#         # ChromeDriver 경로 설정
#         service = Service(r"C:\chromedriver-win64\chromedriver.exe")
#         driver = webdriver.Chrome(service=service, options=chrome_options)
#         driver.get(webpage)

#         # <img> 태그에서 id='artwork__image'인 태그의 src 속성 가져오기
#         image_tag = driver.find_element(By.ID, "artwork__image")
#         image_url = image_tag.get_attribute("src")

#         print(f"Found image URL for {culture}: {image_url}")

#         # SSL 검증 비활성화
#         context = ssl._create_unverified_context()
#         image_response = request.urlopen(image_url, context=context)

#         # 디렉토리 생성
#         culture = culture.replace(",", "").replace("/", " ")
#         download_dir = f"data/met_art/{culture}"
#         os.makedirs(download_dir, exist_ok=True)

#         # 이미지 다운로드 경로 설정 및 다운로드
#         download_path = os.path.join(download_dir, image_url.split("/")[-1])
#         with open(download_path, 'wb') as image_file:
#             image_file.write(image_response.read())

#         # # 이미지 이름에 object_number 또는 culture를 추가해 고유 이름 생성
#         # file_name = f"{culture}_{image_url.split('/')[-1]}"
#         # download_path = os.path.join(download_dir, file_name)

        
#         print(f"Downloaded {download_path}")

#                 # **레이블 저장 (CSV 파일)**
#         with open("labels.csv", "a", newline="", encoding="utf-8") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([culture, webpage, image_url, file_name])

#     except Exception as e:
#         print(f"Error downloading from {webpage} for {culture}: {e}")
    
#     finally:
#         driver.quit()  # 브라우저 닫기

# def main():
#     with open(ARTS_LIST) as f:
#         arts_to_download = [x.strip() for x in f.readlines()]

#     download_count = 0

#     for item in arts_to_download:
#         # Parse the line to get culture and webpage
#         pick = item.split("',")
#         culture = pick[1].replace(" u'", "")
#         webpage = pick[2].replace(" u'", "").replace("')", "")
#         print(f"Processing {culture}: {webpage}")

#         # Selenium을 사용해 이미지 다운로드
#         download_image_selenium(culture, webpage)
#         download_count += 1

#         if 0 < NUMBER_TO_DOWNLOAD == download_count:
#             print("Reached the download limit.")
#             break

#     print("Download completed.")

# if __name__ == "__main__":
#     main()

"""
4. BeautifulSoup만으로는 해당 이미지를 확인할 수 없고 + Selenium으로 페이지를 렌더링해야 이미지 링크를 가져올 수 있습니다.
"""
# # pip install selenium
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from urllib import request
# import os
# import ssl
# from urllib import request



# ARTS_LIST = 'arts-select.list'
# NUMBER_TO_DOWNLOAD = 100

# def download_image_selenium(culture, webpage):
#     try:
#         # ChromeDriver 설정
#         chrome_options = Options()
#         chrome_options.add_argument("--headless")  # Headless 모드 (브라우저 창 없이 실행)
#         chrome_options.add_argument("--no-sandbox")
        
#         # ChromeDriver 경로 설정
#         # service = Service("C:/Users/007/Downloads/chromedriver-win64/chromedriver.exe")
#         # service = Service(r"C:\Users\007\Downloads\chromedriver-win64\chromedriver.exe")
#         service = Service(r"C:\chromedriver-win64\chromedriver.exe")


#         driver = webdriver.Chrome(service=service, options=chrome_options)
#         driver.get(webpage)

#         # <img> 태그에서 id='artwork_image'인 태그의 src 속성 가져오기
#         image_tag = driver.find_element(By.ID, "artwork__image")
#         image_url = image_tag.get_attribute("src")

#         print(f"Found image URL for {culture}: {image_url}")

#         # SSL 검증 비활성화                                         ################ 추가
#         context = ssl._create_unverified_context()
#         image_response = request.urlopen(image_url, context=context)

#         # 디렉토리 생성
#         culture = culture.replace(",", "").replace("/", " ")
#         download_dir = f"data/met_art/{culture}"
#         os.makedirs(download_dir, exist_ok=True)

#         # 이미지 다운로드 경로 설정 및 다운로드
#         download_path = os.path.join(download_dir, image_url.split("/")[-1])
#         image_response = request.urlopen(image_url)
#         with open(download_path, 'wb') as image_file:
#             image_file.write(image_response.read())
        
#         print(f"Downloaded {download_path}")

#         driver.quit()  # 브라우저 닫기

#     except Exception as e:
#         print(f"Error downloading from {webpage} for {culture}: {e}")

# def main():
#     with open(ARTS_LIST) as f:
#         arts_to_download = [x.strip() for x in f.readlines()]

#     download_count = 0

#     for item in arts_to_download:
#         # Parse the line to get culture and webpage
#         pick = item.split("',")
#         culture = pick[1].replace(" u'", "")
#         webpage = pick[2].replace(" u'", "").replace("')", "")
#         print(f"Processing {culture}: {webpage}")

#         # Selenium을 사용해 이미지 다운로드
#         download_image_selenium(culture, webpage)
#         download_count += 1

#         if 0 < NUMBER_TO_DOWNLOAD == download_count:
#             print("Reached the download limit.")
#             break

#     print("Download completed.")

# if __name__ == "__main__":
#     main()


"""
3. Requests + Selenium
Selenium을 사용하면 JavaScript가 렌더링한 HTML을 가져올 수 있습니다.
"""
# # pip install selenium
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# import os
# import time


# ARTS_LIST = 'arts-select.list'
# NUMBER_TO_DOWNLOAD = 100

# # ChromeDriver 경로 설정 (ChromeDriver 설치 필요)
# CHROMEDRIVER_PATH = "path/to/chromedriver"  # chromedriver.exe의 경로로 수정

# # Chrome 옵션 설정
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # 백그라운드에서 실행
# chrome_options.add_argument("--no-sandbox")

# def download_image(culture, webpage):
#     try:
#         service = Service(CHROMEDRIVER_PATH)
#         driver = webdriver.Chrome(service=service, options=chrome_options)
#         driver.get(webpage)
#         time.sleep(2)  # 페이지가 로드될 때까지 대기

#         # id='artwork_image'인 <img> 태그 찾기
#         image_tag = driver.find_element(By.ID, "artwork_image")
#         image_url = image_tag.get_attribute("src")
#         print(f"Found image URL for {culture}: {image_url}")

#         # 디렉토리 생성
#         culture = culture.replace(",", "").replace("/", " ")
#         download_dir = f"data/met_art/{culture}"
#         os.makedirs(download_dir, exist_ok=True)

#         # 이미지 다운로드
#         download_path = os.path.join(download_dir, image_url.split("/")[-1])
#         with open(download_path, 'wb') as image_file:
#             image_file.write(request.urlopen(image_url).read())

#         print(f"Downloaded {download_path}")
#         driver.quit()

#     except Exception as e:
#         print(f"Error downloading from {webpage} for {culture}: {e}")

# def main():
#     with open(ARTS_LIST) as f:
#         arts_to_download = [x.strip() for x in f.readlines()]

#     download_count = 0

#     for item in arts_to_download:
#         pick = item.split("',")
#         culture = pick[1].replace(" u'", "")
#         webpage = pick[2].replace(" u'", "").replace("')", "")
#         print(f"Processing {culture}: {webpage}")

#         download_image(culture, webpage)
#         download_count += 1

#         if 0 < NUMBER_TO_DOWNLOAD == download_count:
#             print("Reached the download limit.")
#             break

#     print("Download completed.")

# if __name__ == "__main__":
#     main()


"""
2. BeautifulSoup 방법
"""

# # pip install beautifulsoup4

# from bs4 import BeautifulSoup
# from urllib import parse, request
# import os
# import errno

# ARTS_LIST = 'arts-select.list'
# NUMBER_TO_DOWNLOAD = 100  # -1이면 모든 이미지를 다운로드

# def download_image(culture, webpage):
#     try:
#         response = request.urlopen(webpage)
#         html_page = response.read().decode('utf-8')
#         soup = BeautifulSoup(html_page, 'html.parser')

#         # <img> 태그에서 이미지 URL 추출 (id가 'artwork_image'인 태그 찾기)
#         image_tag = soup.find('img', {'id': 'artwork_image'})

#         if image_tag and image_tag.get('src'):
#             image_url = image_tag['src']
#             print(f"Found image URL for {culture}: {image_url}")

#             # 디렉토리 생성 (culture 이름을 폴더로 사용)
#             culture = culture.replace(",", "").replace("/", " ")
#             download_dir = f"data/met_art/{culture}"
#             os.makedirs(download_dir, exist_ok=True)

#             # 이미지 다운로드 경로
#             download_path = os.path.join(download_dir, image_url.split("/")[-1])

#             # 이미지 다운로드
#             image_response = request.urlopen(image_url)
#             with open(download_path, 'wb') as image_file:
#                 image_file.write(image_response.read())

#             print(f"Downloaded {download_path}")

#         else:
#             print(f"No image URL found for {culture}, skipping: {webpage}")

#     except Exception as e:
#         print(f"Error downloading from {webpage} for {culture}: {e}")

# def main():
#     with open(ARTS_LIST) as f:
#         arts_to_download = [x.strip() for x in f.readlines()]

#     download_count = 0

#     for item in arts_to_download:
#         # Parse the line to get culture and webpage
#         pick = item.split("',")
#         culture = pick[1].replace(" u'", "")
#         webpage = pick[2].replace(" u'", "").replace("')", "")
#         print(f"Processing {culture}: {webpage}")

#         # 이미지 다운로드
#         download_image(culture, webpage)
#         download_count += 1

#         if 0 < NUMBER_TO_DOWNLOAD == download_count:
#             print("Reached the download limit.")
#             break

#     print("Download completed.")

# if __name__ == "__main__":
#     main()

"""
기존 방법
"""

# from __future__ import print_function
# import os
# import errno
# # from HTMLParser import HTMLParser # Python 2.x에서 사용되었다.
# from html.parser import HTMLParser  # Python 3.x 환경에서 사용한다.
# # import urllib2                    # Python 2.x에서 사용되었다.
# from urllib import parse, request          # Python 3.x 환경에서 사용한다.

# ARTS_LIST = 'arts-select.list'
# NUMBER_TO_DOWNLOAD = 100     # set to -1 to download all

# # The URL for the artifact from Bigquery is a webpage, which contains a link
# # to download the original image.  This class parses for the download link


# class MetArtHTMLParser(HTMLParser):
#     # Look for the image link in the http page and download the original image
#     def handle_starttag(self, tag, attrs):
#         if tag == 'a':
#             # Look for the keyword selectedOrDefaultDownload in an href
#             for attr in attrs:
#                 if (attr[0] == 'href' and
#                         'selectedOrDefaultDownload' in attr[1]):
#                     art_url = attr[1].split("'")[1]
#                     # Return the URL to download the original image
#                     self.data = art_url

# with open(ARTS_LIST) as f:
#     arts_to_download = f.readlines()
#     arts_to_download = [x.strip() for x in arts_to_download]
#     f.close()

# myparser = MetArtHTMLParser()

# for item in arts_to_download:
#     # Parse the line to get the culture label and the webpage for the artifact
#     pick = item.split("',")
#     culture = pick[1].replace(" u'", "")
#     webpage = pick[2].replace(" u'", "").replace("')", "")
#     print(culture, webpage)

#     # Download the webpage and parse for the image URL
#     # response = urllib2.urlopen(webpage)
#     response = request.urlopen(webpage)
    
#     # encoding = response.headers.getparam('charset') 
#     encoding = response.headers.get_param('charset')  # Python 3.x에서는 get_param() 사용

#     html_page = response.read().decode(encoding)

#     try:
#         myparser.feed(html_page)

#         # Create a directory with the culture as name if it doesn't exist yet
#         # (remove characters that are not valid for directory name)
#         culture = culture.replace(",", "")
#         culture = culture.replace("/", " ")
#         download_dir = 'data/met_art/' + culture
#         try:
#             os.makedirs(download_dir)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise

#         # Download the image into the directory
#         download_path = download_dir + '/' + myparser.data.split("/")[-1]
#         image_file = open(download_path, 'wb')
#         # Convert the url to the %-encoded format since it may be
#         # in other format like utf-8
        
#         # image_url = urllib2.quote(myparser.data.encode(encoding), '/:')
#         image_url = parse.quote(myparser.data.encode(encoding), safe='/:')  # quote는 urllib.parse로 대체
        
#         print("image to download:  ", image_url)
        
#         # response = urllib2.urlopen(image_url) 
#         response = request.urlopen(image_url)  # urlopen은 urllib.request로 대체

#         image_file.write(response.read())
#         image_file.close()
#     except:
#         print("Error, skipping url: ", webpage)
