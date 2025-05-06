import pandas as pd
from model_runner import run_model  # 기존에 구현한 함수 사용

# 1. CSV에서 입력 명령어 로드
df = pd.read_csv("intent_input_120.csv", encoding="utf-8-sig")  # 또는 cp949
results = []

# 2. 각 명령어에 대해 run_model 실행
for idx, row in df.iterrows():
    user_input = row["text"]
    print(f"[{idx+1:03}] 실행 중: {user_input}")
    
    try:
        result = run_model(user_input)
        results.append({
            "No": idx + 1,
            "Input": user_input,
            "Output": result["output"],
            "Elapsed_Time": result["elapsed_time"],
            "GPU_Allocated_MB": result["gpu_memory"]["allocated_mb"],
            "GPU_Reserved_MB": result["gpu_memory"]["reserved_mb"]
        })
    except Exception as e:
        results.append({
            "No": idx + 1,
            "Input": user_input,
            "Output": f"ERROR: {e}",
            "Elapsed_Time": None,
            "GPU_Allocated_MB": None,
            "GPU_Reserved_MB": None
        })

# 3. 결과 CSV로 저장
result_df = pd.DataFrame(results)
result_df.to_csv("llm_output_results.csv", index=False, encoding="utf-8-sig")
print("\n✅ 완료! 결과는 'llm_output_results.csv'로 저장됨.")
