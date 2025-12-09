#!/usr/bin/env python3
"""
WandB 오프라인 로그에서 training loss 히스토리를 추출하는 스크립트

사용법:
    python scripts/extract_wandb_history.py wandb/offline-run-20251209_005541-8g230gp0
    python scripts/extract_wandb_history.py wandb/offline-run-20251209_005541-8g230gp0 --output outputs/wandb_history.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("❌ wandb가 설치되지 않았습니다.")
    print("   설치: pip install wandb")


def extract_history_from_offline(run_dir: Path) -> Dict[str, List[Any]]:
    """WandB 오프라인 로그에서 히스토리를 추출합니다."""
    if not HAS_WANDB:
        return {}
    
    history = {
        'train/loss': [],
        'train/learning_rate': [],
        'train/step': [],
        'valid/accuracy': [],
        'valid/best_accuracy': [],
    }
    
    try:
        # WandB API로 오프라인 로그 읽기
        # 오프라인 로그는 로컬 파일 시스템에서 직접 읽을 수 있음
        api = wandb.Api()
        
        # 오프라인 run을 읽기 위해 run ID 추출
        run_id = run_dir.name.split('-')[-1]
        
        # 메타데이터에서 entity와 project 가져오기
        metadata_path = run_dir / "files" / "wandb-metadata.json"
        entity = "schaeck"
        project = "paligemma-xvr-openpi"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                entity = metadata.get('entity', entity)
                project = metadata.get('project', project)
        
        # 온라인 run으로 읽기 시도 (업로드되지 않았으면 실패)
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            print(f"✅ 온라인 run 발견: {run.url}")
            
            # 히스토리 가져오기
            for row in run.scan_history():
                if 'train/loss' in row:
                    history['train/loss'].append({
                        'step': row.get('_step', row.get('train/step', 0)),
                        'value': row['train/loss']
                    })
                if 'train/learning_rate' in row:
                    history['train/learning_rate'].append({
                        'step': row.get('_step', row.get('train/step', 0)),
                        'value': row['train/learning_rate']
                    })
                if 'valid/accuracy' in row:
                    history['valid/accuracy'].append({
                        'step': row.get('_step', 0),
                        'value': row['valid/accuracy']
                    })
            
            print(f"✅ 히스토리 추출 완료:")
            print(f"   train/loss: {len(history['train/loss'])} points")
            print(f"   train/learning_rate: {len(history['train/learning_rate'])} points")
            print(f"   valid/accuracy: {len(history['valid/accuracy'])} points")
            
        except Exception as e:
            print(f"⚠️  온라인 run을 찾을 수 없습니다: {e}")
            print(f"   오프라인 로그는 업로드 후에만 히스토리를 읽을 수 있습니다.")
            print(f"\n💡 대안:")
            print(f"   1. wandb sync로 업로드 후 다시 시도")
            print(f"   2. WandB 바이너리 파일 직접 파싱 (복잡)")
            return {}
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return {}
    
    return history


def save_history_json(history: Dict, output_path: Path):
    """히스토리를 JSON으로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ 히스토리 저장: {output_path}")


def plot_history(history: Dict, output_dir: Path):
    """히스토리를 plot합니다."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib가 없어서 plot을 그릴 수 없습니다.")
        return
    
    if not history.get('train/loss'):
        print("⚠️  Plot할 데이터가 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Training loss
    if history['train/loss']:
        ax = axes[0]
        steps = [h['step'] for h in history['train/loss']]
        losses = [h['value'] for h in history['train/loss']]
        ax.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Validation accuracy
    if history['valid/accuracy']:
        ax = axes[1]
        steps = [h['step'] for h in history['valid/accuracy']]
        accuracies = [h['value'] * 100 for h in history['valid/accuracy']]  # 퍼센트로
        ax.plot(steps, accuracies, 'g-o', markersize=4, linewidth=1.5)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot 저장: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="WandB 오프라인 로그에서 히스토리 추출")
    parser.add_argument("run_dir", help="오프라인 run 디렉토리 경로")
    parser.add_argument("--output", "-o", default="outputs/wandb_history.json",
                       help="출력 JSON 파일 경로")
    parser.add_argument("--plot", action="store_true", help="Plot도 생성")
    
    args = parser.parse_args()
    
    if not HAS_WANDB:
        sys.exit(1)
    
    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {run_path}")
        sys.exit(1)
    
    print(f"📖 오프라인 로그 읽기: {run_path}")
    history = extract_history_from_offline(run_path)
    
    if not history or not any(history.values()):
        print("\n❌ 히스토리를 추출할 수 없습니다.")
        print("   오프라인 로그는 업로드 후에만 히스토리를 읽을 수 있습니다.")
        sys.exit(1)
    
    # JSON 저장
    output_path = Path(args.output)
    save_history_json(history, output_path)
    
    # Plot 생성
    if args.plot:
        plot_history(history, output_path.parent)
    
    print(f"\n✅ 완료!")


if __name__ == "__main__":
    main()

