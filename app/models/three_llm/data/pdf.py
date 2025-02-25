# import os
# import re
# import fitz  # PyMuPDF

# # 텍스트를 청크 단위로 나누는 함수
# def split_text_into_chunks(text, chunk_size=1000):
#     text = re.sub(r'\s+', ' ', text).strip()
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunks, current_chunk = [], ""
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) + 1 < chunk_size:
#             current_chunk += sentence + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# # PDF 파일 처리 (vault 파일에 내용을 추가)
# def process_pdf(file_path, vault_path):
#     with fitz.open(file_path) as doc:
#         text = "\n".join([page.get_text("text") for page in doc])
#     chunks = split_text_into_chunks(text)
    
#     with open(vault_path, "a", encoding="utf-8") as vault_file:
#         # PDF 파일의 이름을 구분자로 기록
#         vault_file.write(f"==== {os.path.basename(file_path)} ====\n")
#         for chunk in chunks:
#             vault_file.write(chunk + "\n")
#         vault_file.write("\n")  # 각 PDF 사이에 빈 줄 추가

# # 현재 스크립트 파일이 위치한 디렉토리를 기준으로 경로 설정
# script_dir = os.path.dirname(os.path.abspath(__file__))
# pdf_dir = os.path.join(script_dir, "RAG_files")

# # 'RAG_files' 폴더가 없으면 생성
# if not os.path.exists(pdf_dir):
#     os.makedirs(pdf_dir)

# # vault 파일의 전체 경로 (모든 PDF 결과를 모아둘 파일)
# vault_path = os.path.join(pdf_dir, "vault.txt")

# # vault 파일이 이미 존재하면 삭제 (새로운 내용으로 덮어쓰기 위해)
# if os.path.exists(vault_path):
#     os.remove(vault_path)

# # PDF 파일만 처리하여 vault 파일에 결과를 추가
# for file in os.listdir(pdf_dir):
#     if file.lower().endswith(".pdf"):
#         file_path = os.path.join(pdf_dir, file)
#         process_pdf(file_path, vault_path)

import os

def merge_txt_files(vault_dir, output_file):
    # vault_dir 내의 모든 txt 파일 목록을 가져옴
    txt_files = [f for f in os.listdir(vault_dir) if f.lower().endswith('.txt')]
    # 정렬해서 파일 순서를 일정하게 유지할 수 있음 (필요 시)
    txt_files.sort()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            file_path = os.path.join(vault_dir, txt_file)
            with open(file_path, 'r', encoding='utf-8') as infile:
                # 파일명을 구분자로 기록하여 어떤 파일의 내용인지 구분 가능하게 함
                outfile.write(f"===== {txt_file} =====\n")
                outfile.write(infile.read())
                outfile.write("\n\n")
    print(f"{len(txt_files)}개의 파일을 '{output_file}'로 병합 완료했습니다.")

if __name__ == '__main__':
    # 현재 스크립트가 위치한 디렉토리를 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vault_dir = os.path.join(script_dir, "vault")
    output_file = os.path.join(script_dir, "merged_vault.txt")
    
    merge_txt_files(vault_dir, output_file)
