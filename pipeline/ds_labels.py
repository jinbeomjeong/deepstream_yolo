def load_labels(path):
    """라벨 파일(한 줄 = 한 클래스)을 읽어 리스트로 반환. 실패 시 빈 리스트."""
    try:
        with open(path) as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"라벨 로드: {path} ({len(labels)}개)")
        return labels
    except FileNotFoundError:
        print(f"[경고] 라벨 파일 없음: {path} — class_id를 숫자로 표시합니다.")
        return []
