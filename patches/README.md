# Big Vision Patches

이 디렉토리에는 `paligemma_jax_training`에서 사용하는 `big_vision` 저장소의 수정사항이 포함되어 있습니다.

## 패치 내용

### `big_vision_paligemma_vs_main.patch`

PaliGemma 모델에 `num_images` 파라미터를 추가하여 동적 이미지 토큰 마스킹을 지원합니다.

**주요 변경사항:**
- `embed_image_and_text` 메서드에 `num_images` 파라미터 추가
- `__call__` 메서드에 `num_images` 파라미터 추가
- `num_images`를 기반으로 `valid_image_mask` 생성하여 패딩된 이미지 토큰 마스킹
- `num_images=0` 케이스 지원 (텍스트 전용 학습)
- `MODEL_DEBUG_EMBEDDING` 및 `MODEL_DEBUG_FORWARD` 환경변수 지원

## 사용 방법

### 1. 패치 적용

```bash
# 방법 1: 스크립트 사용 (권장)
bash patches/apply_big_vision_patch.sh /path/to/big_vision

# 방법 2: 직접 적용
cd /path/to/big_vision
git apply /path/to/paligemma_jax_training/patches/big_vision_paligemma_vs_main.patch
```

### 2. 패치 확인

```bash
cd /path/to/big_vision
git status
git diff big_vision/models/proj/paligemma/paligemma.py
```

### 3. 패치 제거 (필요시)

```bash
cd /path/to/big_vision
git checkout big_vision/models/proj/paligemma/paligemma.py
```

## 주의사항

- 이 패치는 `google-research/big_vision`의 `main` 브랜치를 기준으로 작성되었습니다.
- 다른 버전의 big_vision에 적용할 경우 충돌이 발생할 수 있습니다.
- 패치 적용 후 테스트를 수행하여 정상 동작을 확인하세요.

## 다른 PC로 옮기기

이 `patches/` 디렉토리 전체를 다른 PC로 복사한 후, 위의 사용 방법을 따라 적용하면 됩니다.

