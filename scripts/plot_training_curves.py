#!/usr/bin/env python3
"""
학습 중 저장된 validation_results JSON 파일들로 학습 곡선을 그리는 스크립트

사용법:
    python scripts/plot_training_curves.py outputs/openpi_production/checkpoints
    python scripts/plot_training_curves.py outputs/gqa_openpi/checkpoints --output plots/
    python scripts/plot_training_curves.py outputs/refcocog_openpi/checkpoints --vector-similarity
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib가 설치되지 않았습니다.")
    print("   설치: pip install matplotlib numpy")
    print("   (텍스트 요약만 출력합니다)")


def parse_vector_string(vec_str: str) -> Optional[np.ndarray]:
    """벡터 문자열을 파싱하여 numpy 배열로 변환합니다.
    
    예: "[1.0, 2.0, 3.0, 4.0]" -> np.array([1.0, 2.0, 3.0, 4.0])
    """
    if not isinstance(vec_str, str):
        return None
    
    # 대괄호 제거 및 공백 제거
    vec_str = vec_str.strip()
    if vec_str.startswith('[') and vec_str.endswith(']'):
        vec_str = vec_str[1:-1]
    
    try:
        # 쉼표로 분리하고 숫자로 변환
        numbers = [float(x.strip()) for x in vec_str.split(',') if x.strip()]
        if numbers:
            return np.array(numbers)
    except (ValueError, AttributeError):
        pass
    
    return None


def compute_vector_similarity(pred_vec: np.ndarray, gt_vec: np.ndarray, method: str = 'cosine') -> float:
    """벡터 유사도를 계산합니다.
    
    Args:
        pred_vec: 예측 벡터
        gt_vec: 정답 벡터
        method: 유사도 계산 방법 ('cosine', 'iou', 'normalized_l2')
    
    Returns:
        유사도 점수 (0~1 범위)
    """
    if pred_vec.shape != gt_vec.shape:
        return 0.0
    
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(pred_vec, gt_vec)
        norm_pred = np.linalg.norm(pred_vec)
        norm_gt = np.linalg.norm(gt_vec)
        if norm_pred == 0 or norm_gt == 0:
            return 0.0
        similarity = dot_product / (norm_pred * norm_gt)
        # 0~1 범위로 변환 (cosine similarity는 -1~1)
        return (similarity + 1) / 2
    
    elif method == 'iou' and len(pred_vec) == 4 and len(gt_vec) == 4:
        # IoU for bounding boxes [x, y, width, height]
        # Convert to [x1, y1, x2, y2] format
        pred_x1, pred_y1 = pred_vec[0], pred_vec[1]
        pred_x2, pred_y2 = pred_vec[0] + pred_vec[2], pred_vec[1] + pred_vec[3]
        
        gt_x1, gt_y1 = gt_vec[0], gt_vec[1]
        gt_x2, gt_y2 = gt_vec[0] + gt_vec[2], gt_vec[1] + gt_vec[3]
        
        # Intersection
        inter_x1 = max(pred_x1, gt_x1)
        inter_y1 = max(pred_y1, gt_y1)
        inter_x2 = min(pred_x2, gt_x2)
        inter_y2 = min(pred_y2, gt_y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        pred_area = pred_vec[2] * pred_vec[3]
        gt_area = gt_vec[2] * gt_vec[3]
        union_area = pred_area + gt_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    elif method == 'normalized_l2':
        # Normalized L2 distance (1 - normalized_distance)
        diff = pred_vec - gt_vec
        distance = np.linalg.norm(diff)
        # 정규화를 위해 gt 벡터의 norm 사용
        norm_gt = np.linalg.norm(gt_vec)
        if norm_gt == 0:
            return 1.0 if distance == 0 else 0.0
        normalized_distance = distance / norm_gt
        # 거리를 유사도로 변환 (거리가 작을수록 유사도 높음)
        return max(0.0, 1.0 - normalized_distance)
    
    else:
        # Default: cosine similarity
        return compute_vector_similarity(pred_vec, gt_vec, method='cosine')


def compute_vector_metrics(data: Dict[str, Any], method: str = 'cosine') -> Dict[str, Any]:
    """samples에서 prediction과 ground_truth의 다양한 메트릭을 계산합니다.
    
    Returns:
        Dict with keys:
        - similarity: 평균 유사도 (IoU, cosine, etc.)
        - prediction_variance: prediction들의 분산 (높을수록 다양함, 학습되고 있음을 의미)
        - prediction_gt_correlation: prediction과 ground truth의 상관계수
        - mean_absolute_error: 평균 절대 오차
        - prediction_diversity: prediction들의 다양성 점수
    """
    if 'samples' not in data or not isinstance(data['samples'], list):
        return {}
    
    similarities = []
    pred_vectors = []
    gt_vectors = []
    mae_values = []
    
    for sample in data['samples']:
        if 'prediction' not in sample or 'ground_truth' not in sample:
            continue
        
        pred_vec = parse_vector_string(sample['prediction'])
        gt_vec = parse_vector_string(sample['ground_truth'])
        
        if pred_vec is not None and gt_vec is not None:
            # 유사도 계산
            similarity = compute_vector_similarity(pred_vec, gt_vec, method=method)
            similarities.append(similarity)
            
            # 벡터 저장
            pred_vectors.append(pred_vec)
            gt_vectors.append(gt_vec)
            
            # MAE 계산
            mae = np.mean(np.abs(pred_vec - gt_vec))
            mae_values.append(mae)
    
    if not similarities:
        return {}
    
    metrics = {
        'similarity': np.mean(similarities),
    }
    
    # Prediction 다양성 (분산)
    if pred_vectors:
        pred_array = np.array(pred_vectors)
        # 각 차원별 분산의 평균
        variance_per_dim = np.var(pred_array, axis=0)
        metrics['prediction_variance'] = np.mean(variance_per_dim)
        
        # 전체 벡터 간의 평균 거리 (다양성)
        if len(pred_vectors) > 1:
            distances = []
            for i in range(len(pred_vectors)):
                for j in range(i + 1, len(pred_vectors)):
                    dist = np.linalg.norm(pred_vectors[i] - pred_vectors[j])
                    distances.append(dist)
            metrics['prediction_diversity'] = np.mean(distances) if distances else 0.0
        else:
            metrics['prediction_diversity'] = 0.0
    
    # Prediction-GT 상관관계
    if pred_vectors and gt_vectors:
        pred_array = np.array(pred_vectors)
        gt_array = np.array(gt_vectors)
        
        # 각 차원별 상관계수의 평균
        correlations = []
        for dim in range(pred_array.shape[1]):
            if np.std(pred_array[:, dim]) > 0 and np.std(gt_array[:, dim]) > 0:
                corr = np.corrcoef(pred_array[:, dim], gt_array[:, dim])[0, 1]
                correlations.append(corr)
        metrics['prediction_gt_correlation'] = np.mean(correlations) if correlations else 0.0
    
    # Mean Absolute Error
    if mae_values:
        metrics['mean_absolute_error'] = np.mean(mae_values)
    
    return metrics


def compute_average_vector_similarity(data: Dict[str, Any], method: str = 'cosine') -> Optional[float]:
    """samples에서 prediction과 ground_truth의 평균 벡터 유사도를 계산합니다."""
    metrics = compute_vector_metrics(data, method=method)
    return metrics.get('similarity')


def load_validation_results(checkpoint_dir: Path, use_vector_similarity: bool = False, similarity_method: str = 'cosine') -> List[Dict[str, Any]]:
    """모든 validation_results JSON 파일을 읽습니다."""
    results = []
    
    validation_files = sorted(checkpoint_dir.glob("validation_results_step_*.json"))
    
    for file_path in validation_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Step 번호 추출
            step_str = file_path.stem.replace("validation_results_step_", "")
            try:
                step = int(step_str)
            except ValueError:
                step = None
            
            # 메트릭 추출
            metrics = {
                'step': step,
                'file': file_path.name,
            }
            
            # 벡터 유사도 모드인 경우
            if use_vector_similarity:
                vector_metrics = compute_vector_metrics(data, method=similarity_method)
                if vector_metrics:
                    metrics['accuracy'] = vector_metrics.get('similarity', 0.0)
                    metrics['vector_similarity'] = vector_metrics.get('similarity', 0.0)
                    # 학습 여부 확인을 위한 추가 메트릭
                    metrics['prediction_variance'] = vector_metrics.get('prediction_variance', 0.0)
                    metrics['prediction_gt_correlation'] = vector_metrics.get('prediction_gt_correlation', 0.0)
                    metrics['mean_absolute_error'] = vector_metrics.get('mean_absolute_error', 0.0)
                    metrics['prediction_diversity'] = vector_metrics.get('prediction_diversity', 0.0)
            else:
                # summary 안에 accuracy가 있을 수 있음
                if 'summary' in data and isinstance(data['summary'], dict):
                    if 'accuracy' in data['summary']:
                        metrics['accuracy'] = data['summary']['accuracy']
                elif 'accuracy' in data:
                    metrics['accuracy'] = data['accuracy']
            
            # best_accuracy는 별도로 추적
            if 'best_accuracy' in data:
                metrics['best_accuracy'] = data['best_accuracy']
            elif 'accuracy' in metrics:
                # best_accuracy가 없으면 현재 accuracy를 best로 사용
                metrics['best_accuracy'] = metrics['accuracy']
            
            results.append(metrics)
        except Exception as e:
            print(f"⚠️  파일 읽기 실패 {file_path}: {e}")
    
    return sorted(results, key=lambda x: x.get('step', 0) if x.get('step') else 0)


def plot_curves(results: List[Dict], output_dir: Path, use_vector_similarity: bool = False, similarity_method: str = 'cosine'):
    """학습 곡선을 plot합니다."""
    metric_name = 'Vector Similarity' if use_vector_similarity else 'Accuracy'
    
    if not HAS_MATPLOTLIB:
        print("\n📊 텍스트 요약:")
        print("-" * 60)
        for r in results:
            step = r.get('step', 'N/A')
            acc = r.get('accuracy', 'N/A')
            best = r.get('best_accuracy', 'N/A')
            if isinstance(acc, (int, float)):
                acc = f"{acc*100:.2f}%"
            if isinstance(best, (int, float)):
                best = f"{best*100:.2f}%"
            step_str = str(step) if step != 'N/A' else 'N/A'
            print(f"  Step {step_str:>6s}: {metric_name}={str(acc):>8s}, Best={str(best):>8s}")
            if use_vector_similarity:
                var = r.get('prediction_variance', 'N/A')
                corr = r.get('prediction_gt_correlation', 'N/A')
                if isinstance(var, (int, float)):
                    var = f"{var:.2f}"
                if isinstance(corr, (int, float)):
                    corr = f"{corr:.3f}"
                print(f"           Variance={str(var):>8s}, Correlation={str(corr):>8s}")
        return
    
    steps = [r.get('step', 0) for r in results if r.get('step')]
    accuracies = [r.get('accuracy', 0) for r in results if 'accuracy' in r]
    best_accuracies = [r.get('best_accuracy', 0) for r in results if 'best_accuracy' in r]
    
    if not steps or not accuracies:
        print("❌ Plot할 데이터가 없습니다.")
        return
    
    # Accuracy를 퍼센트로 변환 (벡터 유사도는 이미 0~1 범위이므로 변환)
    if accuracies:
        if use_vector_similarity:
            # 벡터 유사도는 이미 0~1 범위이므로 퍼센트로 변환
            accuracies = [a * 100 for a in accuracies]
            best_accuracies = [b * 100 for b in best_accuracies] if best_accuracies else []
        elif accuracies[0] < 1.0:
            # 일반 accuracy도 0~1 범위라면 퍼센트로 변환
            accuracies = [a * 100 for a in accuracies]
            best_accuracies = [b * 100 for b in best_accuracies] if best_accuracies else []
    
    if use_vector_similarity and any('prediction_variance' in r for r in results):
        # 벡터 유사도 모드이고 추가 메트릭이 있는 경우: subplot 사용
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Similarity (IoU/Cosine) plot
        ax1 = axes[0, 0]
        label = f'Validation {metric_name} ({similarity_method})'
        ax1.plot(steps, accuracies, 'b-o', label=label, markersize=4, linewidth=1.5)
        if best_accuracies:
            best_label = f'Best {metric_name}'
            ax1.plot(steps, best_accuracies, 'r--s', label=best_label, markersize=4, linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel(f'{metric_name} (%)', fontsize=11)
        ax1.set_title(f'Validation {metric_name} ({similarity_method.upper()})', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # 2. Prediction Variance (다양성) - 높을수록 학습되고 있음
        ax2 = axes[0, 1]
        variances = [r.get('prediction_variance', 0) for r in results if 'prediction_variance' in r]
        var_steps = [r.get('step', 0) for r in results if 'prediction_variance' in r]
        if variances:
            ax2.plot(var_steps, variances, 'g-o', label='Prediction Variance', markersize=4, linewidth=1.5)
            ax2.set_xlabel('Step', fontsize=11)
            ax2.set_ylabel('Variance', fontsize=11)
            ax2.set_title('Prediction Diversity (Higher = More Learning)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
        
        # 3. Prediction-GT Correlation (상관관계) - 높을수록 학습되고 있음
        ax3 = axes[1, 0]
        correlations = [r.get('prediction_gt_correlation', 0) for r in results if 'prediction_gt_correlation' in r]
        corr_steps = [r.get('step', 0) for r in results if 'prediction_gt_correlation' in r]
        if correlations:
            ax3.plot(corr_steps, correlations, 'm-o', label='Prediction-GT Correlation', markersize=4, linewidth=1.5)
            ax3.set_xlabel('Step', fontsize=11)
            ax3.set_ylabel('Correlation', fontsize=11)
            ax3.set_title('Prediction-Ground Truth Correlation (Higher = More Learning)', fontsize=12, fontweight='bold')
            ax3.set_ylim([-1, 1])
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best')
        
        # 4. Mean Absolute Error (MAE) - 낮을수록 좋음
        ax4 = axes[1, 1]
        mae_values = [r.get('mean_absolute_error', 0) for r in results if 'mean_absolute_error' in r]
        mae_steps = [r.get('step', 0) for r in results if 'mean_absolute_error' in r]
        if mae_values:
            ax4.plot(mae_steps, mae_values, 'c-o', label='Mean Absolute Error', markersize=4, linewidth=1.5)
            ax4.set_xlabel('Step', fontsize=11)
            ax4.set_ylabel('MAE', fontsize=11)
            ax4.set_title('Mean Absolute Error (Lower = Better)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best')
        
        plt.tight_layout()
        
    else:
        # 일반 모드: 단일 plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Accuracy plot
        label = f'Validation {metric_name}' if use_vector_similarity else 'Validation Accuracy'
        ax.plot(steps, accuracies, 'b-o', label=label, markersize=4, linewidth=1.5)
        
        # Best accuracy plot
        if best_accuracies:
            best_label = f'Best {metric_name}' if use_vector_similarity else 'Best Accuracy'
            ax.plot(steps, best_accuracies, 'r--s', label=best_label, markersize=4, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Step', fontsize=12)
        ylabel = f'{metric_name} (%)' if use_vector_similarity else 'Accuracy (%)'
        ax.set_ylabel(ylabel, fontsize=12)
        title = f'Training Progress - Validation {metric_name}' if use_vector_similarity else 'Training Progress - Validation Accuracy'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 최종 값 표시
        if accuracies:
            final_acc = accuracies[-1]
            final_step = steps[-1]
            ax.annotate(f'Final: {final_acc:.2f}%', 
                       xy=(final_step, final_acc),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot 저장: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="학습 곡선 plot 생성")
    parser.add_argument("checkpoint_dir", help="Checkpoint 디렉토리 경로")
    parser.add_argument("--output", "-o", default="outputs/plots", 
                       help="출력 디렉토리 (기본: outputs/plots)")
    parser.add_argument("--vector-similarity", action="store_true",
                       help="벡터 유사도 모드 사용 (prediction과 ground_truth가 벡터 형태인 경우)")
    parser.add_argument("--similarity-method", choices=['cosine', 'iou', 'normalized_l2'],
                       default='cosine',
                       help="벡터 유사도 계산 방법 (기본: cosine)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {checkpoint_path}")
        sys.exit(1)
    
    print(f"📖 Validation results 읽기: {checkpoint_path}")
    if args.vector_similarity:
        print(f"   벡터 유사도 모드: {args.similarity_method}")
    results = load_validation_results(checkpoint_path, 
                                     use_vector_similarity=args.vector_similarity,
                                     similarity_method=args.similarity_method)
    
    if not results:
        print("❌ Validation results 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    print(f"✅ {len(results)}개의 validation 결과를 찾았습니다.")
    
    output_dir = Path(args.output)
    print(f"📊 Plot 생성 중...")
    plot_curves(results, output_dir, 
                use_vector_similarity=args.vector_similarity,
                similarity_method=args.similarity_method)
    
    print(f"\n✅ 완료! 출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()

