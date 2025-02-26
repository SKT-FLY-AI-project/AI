import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

import re
import torch
from sentence_transformers import SentenceTransformer, util

# pip install langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from sentence_transformers import SentenceTransformer, util


def create_and_save_vector_store(file_path="Art_RAG.txt", save_path=None):
    """미술 텍스트 데이터를 로드하고 벡터 스토어를 생성한 후 저장하는 함수"""
    # 현재 스크립트 파일 위치 기준으로 데이터 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    absolute_file_path = os.path.join(data_dir, file_path)
    
    if save_path is None:
        save_path = os.path.join(data_dir, "vector_store")
    
    print(f"현재 스크립트 디렉토리: {current_dir}")
    print(f"데이터 디렉토리: {data_dir}")
    print(f"파일 경로: {absolute_file_path}")
    print(f"저장 경로: {save_path}")
    
    # 파일 존재 확인
    if not os.path.exists(absolute_file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {absolute_file_path}")
    
    # 파일 로드
    print(f"파일을 로드합니다: {absolute_file_path}")
    try:
        loader = TextLoader(absolute_file_path, encoding="utf-8")
        print("TextLoader 생성 완료")
        
        documents = loader.load()
        print(f"문서 로드 완료: {len(documents)} 문서")

        # 청크 분할
        print("청크 분할 시작...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 더 작은 청크 사이즈로 조정
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"청크 분할 완료: {len(chunks)} 청크 생성")
        
        # 임베딩 모델 로드
        print("임베딩 모델 로드 시작...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'}
        )
        print("임베딩 모델 로드 완료")

        # FAISS 인덱스 생성
        print("FAISS 인덱스 생성 시작...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        print("FAISS 인덱스 생성 완료")
        
        # 벡터 스토어 저장
        print(f"벡터 스토어 저장 시작: {save_path}")
        vector_store.save_local(save_path)
        print("벡터 스토어 저장 완료")

        return vector_store
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print(f"오류 유형: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        return None


def load_vector_store(save_path=None):
    """저장된 벡터 스토어를 로드하는 함수"""
    # 현재 스크립트 파일 위치 기준으로 저장 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    if save_path is None:
        save_path = os.path.join(data_dir, "vector_store")
    
    print(f"벡터 스토어 로드 경로: {save_path}")
    
    try:
        # 저장 경로 존재 확인
        if not os.path.exists(save_path):
            print(f"저장된 벡터 스토어를 찾을 수 없습니다: {save_path}")
            print("새 벡터 스토어를 생성합니다...")
            return create_and_save_vector_store(save_path=save_path)
        
        # 임베딩 모델 로드
        print("임베딩 모델 로드 시작...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'}
        )
        print("임베딩 모델 로드 완료")
        
        # 벡터 스토어 로드
        print("벡터 스토어 로드 시작...")
        vector_store = FAISS.load_local(save_path, embedding_model)
        print("벡터 스토어 로드 완료")
        
        return vector_store
        
    except Exception as e:
        print(f"벡터 스토어 로드 중 오류 발생: {e}")
        print(f"오류 유형: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        return None
    
# # 초기 벡터 스토어 생성을 위한 스크립트 (별도 실행)
# if __name__ == "__main__":
#     # 이 부분은 별도 스크립트로 한 번만 실행하여 벡터 스토어를 생성하는 용도
#     print("벡터 스토어 생성 및 저장 스크립트 시작")
#     create_and_save_vector_store()
#     print("벡터 스토어 생성 및 저장 완료")


import re
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

# 전처리 및 문장 분리 함수
def preprocess_art_rag(file_path):
    """
    Art_RAG.txt 파일을 읽어와 불필요한 공백을 정리한 후,
    문장별로 분리하여 리스트로 반환합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 앞뒤 공백 제거 및 다중 공백을 하나의 공백으로 치환
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # 문장 분리: 마침표, 물음표, 느낌표 뒤의 공백을 기준으로 분리
    sentences = re.split(r'(?<=[.?!])\s+', text)
    # 빈 문장 제거
    sentences = [s for s in sentences if s]
    return sentences

# 문장별 인덱스 생성 및 임베딩 계산 함수
def build_sentence_embeddings(sentences, model):
    """
    입력된 문장 리스트에 대해 1부터 시작하는 인덱스 딕셔너리와
    각 문장에 대한 임베딩(텐서)을 계산하여 반환합니다.
    """
    # 1부터 시작하는 문자열 인덱스를 키로 하는 딕셔너리 생성
    sentence_dict = {str(idx): sentence for idx, sentence in enumerate(sentences, start=1)}
    # 전체 문장에 대한 임베딩 계산 (Tensor 형태)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return sentence_dict, embeddings

# 전처리 및 임베딩 데이터를 저장하는 함수
def save_precomputed_data(file_path, output_pickle='precomputed_data.pkl'):
    """
    Art_RAG.txt 파일을 읽어 전처리 및 임베딩 계산 후,
    결과를 pickle 파일로 저장합니다.
    """
    # 파일 읽기 및 문장 분리
    sentences = preprocess_art_rag(file_path)
    # 임베딩 모델 로드 (KR-SBERT)
    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    sentence_dict, embeddings = build_sentence_embeddings(sentences, model)
    # 저장할 데이터 구성
    precomputed_data = {
        "sentence_dict": sentence_dict,
        "embeddings": embeddings
    }
    # pickle로 저장
    with open(output_pickle, 'wb') as f:
        pickle.dump(precomputed_data, f)
    print(f"전처리 및 임베딩 데이터가 '{output_pickle}' 파일로 저장되었습니다.")

# 전처리된 데이터를 불러오는 함수
def load_precomputed_data(pickle_file='precomputed_data.pkl'):
    """
    저장된 pickle 파일에서 전처리된 데이터를 불러와 반환합니다.
    """
    with open(pickle_file, 'rb') as f:
        precomputed_data = pickle.load(f)
    return precomputed_data

# 예시: 전처리 및 저장
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    art_rag_file = os.path.join(current_dir, "data", "Art_RAG.txt")
    save_precomputed_data(art_rag_file)