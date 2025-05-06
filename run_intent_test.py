import pandas as pd
from intent_classifier import classify_text

# 입력 CSV 로드
input_df = pd.read_csv("intent_input_120.csv")
results = []

# 각 입력 문장 분류
for idx, row in input_df.iterrows():
    text = row["text"]
    result = classify_text(text)
    results.append({
        "No": idx + 1,
        "Input": text,
        "Predicted_Label": result["label"],
        "Score": result["score"]
    })
    print(f"[{idx+1:03}] {text} → {result['label']} (score={result['score']})")

# 결과 저장
df = pd.DataFrame(results)
df.to_csv("intent_test_results.csv", index=False, encoding="utf-8-sig")
print("\n✅ 테스트 완료! 결과는 'intent_test_results.csv'에 저장됨.")
