# PaliGemma Fine-tuning on XVR

PaliGemma 모델을 XVR 데이터셋으로 fine-tuning하는 프로젝트.  
OpenPI의 pi0.5_base 체크포인트(로봇 데이터 10k+ 시간 사전학습)를 시작점으로 사용.

## 📋 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. XVR 학습 (이 프로젝트)                                                 │
│    OpenPI pi0.5_base → XVR 학습 → 학습된 PaliGemma 체크포인트            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. OpenPI 통합 및 LIBERO 평가                                            │
│    학습된 체크포인트 → OpenPI 형식 변환 → LIBERO 벤치마크 실행            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: XVR 학습

### Quick Start (Vast.ai)

#### 1. GCP에 파일 업로드 (한 번만)

```bash
# 체크포인트 (5.5GB)
gsutil cp checkpoints/pi05_base_paligemma.npz gs://YOUR_BUCKET/checkpoints/

# 토크나이저 (4MB)
gsutil cp assets/paligemma_tokenizer.model gs://YOUR_BUCKET/assets/

# XVR 데이터셋
gsutil -m rsync -r ../XVR gs://YOUR_BUCKET/data/XVR
```

#### 2. Vast.ai 인스턴스 설정

**권장 사양:**
- GPU: NVIDIA L40S (48GB) 또는 A100 (40GB+)
- VRAM: 40GB+ (XVR 학습만 하면 24GB도 가능)
- Disk: 100GB+
- Image: **CUDA 12.x + Python 3.10** (JAX 기반이므로 PyTorch 이미지 아니어도 됨)

**LIBERO 평가까지 하려면:**
- Docker 지원 필요 (대부분 Vast.ai 인스턴스는 지원)
- X11 forwarding 또는 headless 렌더링 (EGL/OSMesa)

**On-start Script:**
```bash
cd /workspace
git clone https://github.com/SChaeck/paligemma_jax_training.git
cd paligemma_jax_training
GCP_BUCKET=gs://riselab-xvr-us ./setup.sh
```

**참고:** `YOUR_BUCKET`을 실제 GCP 버킷 이름으로 변경하세요.

#### 3. 학습

```bash
conda activate paligemma_training

# W&B 로그인 (선택사항, 로깅을 원하는 경우)
wandb login

# Quick overfit test (5분, 파이프라인 확인용)
python scripts/01_overfit_test.py

# Full training
python scripts/03_train_production.py --env envs/.env.openpi
```

**⚠️ Vast.ai 체크리스트:**

- [ ] GCP 버킷에 체크포인트/데이터 업로드 완료
- [ ] `setup.sh` 실행 완료 (conda 환경 생성, 의존성 설치)
- [ ] `conda activate paligemma_training` 성공
- [ ] `python scripts/01_overfit_test.py` 성공 (파이프라인 확인)
- [ ] W&B 로그인 완료 (`wandb login`) - 선택사항
- [ ] `.env` 파일에 하이퍼파라미터 설정 완료
- [ ] `python scripts/03_train_production.py` 실행 중

#### 4. 학습 결과 확인

```bash
# 학습된 체크포인트 위치
ls outputs/*/checkpoints/

# 평가
python -c "
from src.config import load_config
from src.model import load_paligemma_model
from src.eval import evaluate_accuracy

config = load_config('envs/.env.openpi')
config.model.checkpoint_path = 'outputs/production/checkpoints/checkpoint_final.npz'
# ... 평가 코드
"
```

---

## Part 2: OpenPI 통합 및 LIBERO 평가

XVR로 학습된 PaliGemma를 OpenPI에 통합하고 LIBERO 벤치마크로 평가합니다.

### 사전 요구사항

```bash
# OpenPI 저장소 (이미 있다면 스킵)
cd /home/suchae/pi_workspace
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git submodule update --init --recursive

# 환경 설정
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e packages/openpi-client
```

### Step 1: 학습된 체크포인트를 OpenPI 형식으로 변환

XVR 학습된 체크포인트 (.npz)를 OpenPI가 읽을 수 있는 위치에 복사:

```bash
# 학습된 체크포인트를 OpenPI에서 접근 가능한 위치로 복사
cp /path/to/paligemma_jax_training/outputs/production/checkpoints/checkpoint_final.npz \
   /home/suchae/pi_workspace/openpi/xvr_trained_paligemma.npz
```

### Step 2: 커스텀 Weight Loader 생성

OpenPI에서 XVR 학습된 PaliGemma를 로드하는 커스텀 로더 추가:

```python
# openpi/src/openpi/training/weight_loaders.py 에 추가

@dataclasses.dataclass(frozen=True)
class XVRTrainedPaliGemmaLoader(WeightLoader):
    """Loads XVR-trained PaliGemma weights.
    
    This replaces the PaliGemma weights in pi0.5_base with XVR-trained weights.
    Action expert and other components remain unchanged.
    """
    checkpoint_path: str = "./xvr_trained_paligemma.npz"
    
    def load(self, params: at.Params) -> at.Params:
        # First, load the pi05_base checkpoint for action expert weights
        base_params = _model.restore_params(
            download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"),
            restore_type=np.ndarray
        )
        
        # Then, load XVR-trained PaliGemma weights
        with open(self.checkpoint_path, "rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        
        # Convert to OpenPI format (params/... → PaliGemma/...)
        xvr_params = {}
        for k, v in flat_params.items():
            if k.startswith("params/"):
                new_key = "PaliGemma/" + k[7:]  # Remove "params/" prefix
                xvr_params[new_key] = v
        
        xvr_loaded = flax.traverse_util.unflatten_dict(xvr_params, sep="/")
        
        # Merge: XVR PaliGemma + base action expert
        merged = _merge_params(xvr_loaded, base_params, missing_regex=".*")
        return merged
```

### Step 3: LIBERO 벤치마크용 학습 Config 생성

```python
# openpi/src/openpi/training/config.py 에 추가

TrainConfig(
    name="pi05_xvr_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    # XVR 학습된 PaliGemma 사용
    weight_loader=weight_loaders.XVRTrainedPaliGemmaLoader("./xvr_trained_paligemma.npz"),
    num_train_steps=30_000,
),
```

### Step 4: LIBERO Fine-tuning 실행

```bash
cd /home/suchae/pi_workspace/openpi

# LIBERO 데이터로 fine-tuning (XVR 학습된 PaliGemma 기반)
uv run scripts/train.py --config pi05_xvr_libero
```

### Step 5: LIBERO 벤치마크 평가

```bash
cd /home/suchae/pi_workspace/openpi

# Docker로 LIBERO 벤치마크 실행
sudo xhost +local:docker

# XVR 학습된 체크포인트로 평가
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_xvr_libero --policy.dir ./checkpoints/pi05_xvr_libero/YOUR_STEP/params" \
docker compose -f examples/libero/compose.yml up --build
```

### 예상 결과 비교

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|----------------|---------------|-------------|-----------|---------|
| π0.5 base (baseline) | ? | ? | ? | ? | ? |
| π0.5 + XVR (ours) | ? | ? | ? | ? | ? |
| π0.5 @ 30k (OpenPI 공식) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

---

## 프로젝트 구조

```
paligemma_jax_training/
├── assets/                    # 토크나이저 등 모델 에셋
│   └── paligemma_tokenizer.model
├── checkpoints/               # 모델 체크포인트
│   └── pi05_base_paligemma.npz  # OpenPI pi0.5_base에서 추출 (5.5GB)
├── envs/                      # 환경 설정 프리셋
│   ├── .env.openpi           # OpenPI 체크포인트 사용 (권장)
│   ├── .env.overfit          # 오버핏 테스트용
│   └── .env.production       # 프로덕션 학습
├── scripts/                   # 학습 스크립트
│   ├── 01_overfit_test.py    # 빠른 오버핏 테스트
│   └── 03_train_production.py # 프로덕션 학습
├── src/                       # 소스 코드
├── outputs/                   # 학습 결과물 (gitignore)
├── setup.sh                   # 환경 설정 스크립트
└── requirements.txt           # Python 의존성
```

## 환경 변수 및 하이퍼파라미터

주요 환경 변수 (`envs/.env.*` 파일에서 설정):

### 모델 설정
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_CHECKPOINT_PATH` | `./checkpoints/pi05_base_paligemma.npz` | 체크포인트 경로 |
| `MODEL_TOKENIZER_PATH` | `./assets/paligemma_tokenizer.model` | 토크나이저 경로 |

### 데이터 설정
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATA_BASE_DIR` | `../XVR` | XVR 데이터 디렉토리 |
| `MAX_SEQ_LENGTH` | `256` | 최대 시퀀스 길이 |

### 학습 하이퍼파라미터
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TRAINABLE_PARAMS` | `attention_only` | 학습할 파라미터 (`attention_only`, `all`) |
| `BATCH_SIZE` | `8` | 배치 크기 |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Gradient accumulation (effective batch = batch_size × accumulation_steps) |
| `LEARNING_RATE` | `0.03` | 학습률 |
| `NUM_EPOCHS` | `10` | 에포크 수 |
| `WARMUP_PERCENT` | `0.10` | Warmup 비율 (전체 steps의 %) |
| `LR_SCHEDULE` | `cosine` | LR 스케줄 (`cosine`, `constant`, `linear`) |
| `MAX_GRAD_NORM` | `1.0` | Gradient clipping |
| `PRECISION` | `float32` | Precision (`float32`, `bfloat16`, `float16`) |

### 로깅 및 체크포인트
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `USE_WANDB` | `false` | Weights & Biases 사용 여부 |
| `WANDB_PROJECT` | `paligemma-xvr` | W&B 프로젝트 이름 |
| `WANDB_ENTITY` | (없음) | W&B 엔티티 (없으면 개인 계정) |
| `LOG_EVERY` | `10` | 로그 출력 주기 (steps) |
| `EVAL_EVERY` | `100` | 평가 주기 (steps) |
| `CHECKPOINT_EVERY` | `500` | 체크포인트 저장 주기 (steps) |
| `MAX_CHECKPOINTS_TO_KEEP` | `3` | 유지할 체크포인트 개수 |

### W&B 설정 예시

`.env` 파일에 추가:
```bash
USE_WANDB=true
WANDB_PROJECT=paligemma-xvr
WANDB_ENTITY=your-team-name  # 선택사항, 없으면 개인 계정
```

학습 시작 전 W&B 로그인:
```bash
wandb login
```

## setup.sh 옵션

```bash
./setup.sh [options]

Options:
  --gcp-bucket BUCKET   GCP 버킷 URL (예: gs://my-bucket)
  --skip-data           XVR 데이터 다운로드 스킵
  --skip-checkpoint     체크포인트 다운로드 스킵
  --use-kaggle          OpenPI 대신 Kaggle PaliGemma 사용
```

## OpenPI vs Vanilla PaliGemma

| 항목 | Vanilla PaliGemma | OpenPI pi0.5_base |
|------|-------------------|-------------------|
| 파일 크기 | 5.5GB | 5.5GB |
| 사전학습 | 이미지 캡셔닝 | 로봇 조작 (10k+ 시간) |
| 파라미터 | 동일 구조 | **값이 다름** |
| 권장 용도 | 일반 VQA | 로봇/액션 태스크 |

## Requirements

- Python 3.10
- CUDA 12.x
- JAX 0.4.30
- Flax 0.8.4

## License

MIT
