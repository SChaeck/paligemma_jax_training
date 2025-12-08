#!/bin/bash
# big_vision 패치 적용 스크립트
# 
# 이 스크립트는 paligemma_jax_training에서 사용하는 big_vision 수정사항을 적용합니다.
# num_images 파라미터를 추가하여 동적 이미지 토큰 마스킹을 지원합니다.
#
# 사용법: 
#   ./apply_big_vision_patch.sh <big_vision_경로>
#   또는
#   bash patches/apply_big_vision_patch.sh /path/to/big_vision

if [ -z "$1" ]; then
    echo "사용법: $0 <big_vision_경로>"
    echo ""
    echo "예시:"
    echo "  $0 ../big_vision"
    echo "  $0 /home/user/big_vision"
    exit 1
fi

BIG_VISION_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/big_vision_paligemma_vs_main.patch"

if [ ! -d "$BIG_VISION_PATH" ]; then
    echo "오류: $BIG_VISION_PATH 디렉토리가 없습니다."
    exit 1
fi

if [ ! -f "$PATCH_FILE" ]; then
    echo "오류: 패치 파일을 찾을 수 없습니다: $PATCH_FILE"
    exit 1
fi

echo "패치 파일: $PATCH_FILE"
echo "적용 대상: $BIG_VISION_PATH"
echo ""

cd "$BIG_VISION_PATH"

# git 저장소인지 확인
if [ ! -d ".git" ]; then
    echo "경고: $BIG_VISION_PATH는 git 저장소가 아닙니다."
    echo "패치를 적용하려면 git 저장소여야 합니다."
    exit 1
fi

# 패치 적용
git apply "$PATCH_FILE"
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 패치 적용 성공!"
    echo ""
    echo "변경된 파일:"
    git status --short
else
    echo ""
    echo "✗ 패치 적용 실패. 충돌이 있을 수 있습니다."
    echo "수동으로 확인해주세요."
    exit 1
fi
