from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 라벨 ID -> 라벨 이름 매핑 (선택)
id2label = {
    "LABEL_0": "open_teaching",
    "LABEL_1": "inspection_window",
    "LABEL_2": "start_inspection",
    "LABEL_3": "calibration",
    "LABEL_4": "setting",
    "LABEL_5": "change_recipe"
}

# 분류 모델 체크포인트 경로 (학습 후 저장된 경로)
MODEL_DIR = "./result/checkpoint-40"

# 토크나이저 및 모델 로딩
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 분류 파이프라인 생성 (GPU 사용 시 device=0, CPU는 -1)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)


def classify_text(text: str) -> dict:
    result = classifier(text)[0]
    label = id2label.get(result["label"], result["label"])
    score = round(result["score"], 4)
    return {
        "label": label,
        "score": score
    }
