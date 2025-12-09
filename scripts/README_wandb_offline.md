# WandB 오프라인 로그에서 Training Loss 보기

## 문제
WandB 오프라인 로그의 바이너리 파일(.wandb)에서 training loss 히스토리를 직접 추출하는 것은 복잡합니다.

## 해결 방법

### 방법 1: WandB 업로드 후 온라인에서 확인 (권장)
```bash
# 업로드
wandb sync wandb/offline-run-20251209_005541-8g230gp0 --entity schaeck

# 웹에서 확인
# https://wandb.ai/schaeck/paligemma-xvr-openpi/runs/8g230gp0
```

### 방법 2: Summary에서 최종 값 확인
```bash
python scripts/view_wandb_summary.py wandb/offline-run-20251209_005541-8g230gp0
```

### 방법 3: 학습 중 별도 로그 저장 (향후 학습용)
학습 스크립트를 수정하여 training loss를 JSON으로 저장:
```python
# training_loss_history.json에 저장
loss_history = {
    'steps': [],
    'losses': [],
    'learning_rates': []
}
```

### 방법 4: WandB 바이너리 파일 직접 파싱 (고급)
WandB는 protobuf 형식을 사용하므로, `wandb` 라이브러리가 필요합니다:
```bash
pip install wandb
python scripts/extract_wandb_history.py wandb/offline-run-20251209_005541-8g230gp0 --plot
```

## 현재 가능한 것
- ✅ Validation accuracy 히스토리 (validation_results JSON에서)
- ✅ 최종 training loss (WandB summary에서)
- ❌ Training loss 히스토리 (업로드 필요 또는 별도 저장 필요)

## 권장 사항
**향후 학습부터는 training loss를 별도 JSON 파일로 저장하는 것을 권장합니다.**

