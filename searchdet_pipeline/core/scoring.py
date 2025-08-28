#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from ..utils.config import ScoringConfig

# Простые, устойчивые утилиты (как в рабочем скрипте)
def _to_matrix(X: np.ndarray, D: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X.reshape(0, (D if D is not None else 0))
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if D is not None and X.shape[1] != D:
        raise ValueError(f"Dim mismatch: expected D={D}, got {X.shape}")
    return X

def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = _to_matrix(A)
    B = _to_matrix(B, A.shape[1]) if B.size else _to_matrix(B)
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)

    # предполагаем, что входы уже L2-нормированы (мы так и делаем в embeddings),
    # но всё равно нормализуем на всякий случай
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    sims = A @ B.T
    sims = np.clip(sims, -1.0, 1.0).astype(np.float32)
    return sims

class ScoreCalculator:
    """
    Упрощённый и предсказуемый скорер:
    - pos_score = 0.7*max_cos + 0.3*mean_topk_cos  (далее перевод в [0,1])
    - neg_score = max_cos_neg → [0,1]
    - решение: (pos01 >= min_pos) AND (pos01 - neg01 >= thr) AND (n <= neg_cap)
    - мультикласс: берём лучший класс по pos01, разделение с аргументом class_separation.
    Никакой агрессивной квантильной отсечки, совпадает духом с твоим рабочим скриптом.
    """

    def __init__(self, detector, params: Dict[str, Any] | None = None, config: ScoringConfig | None = None):
        self.detector = detector
        if config is not None:
            self.config = config
        else:
            params = params or {}
            self.config = ScoringConfig(
                min_pos_score=float(params.get('min_pos_score', 0.35)),     # мягкий дефолт
                decision_threshold=float(params.get('decision_threshold', 0.06)),
                class_separation=float(params.get('class_separation', 0.04)),
                neg_cap=float(params.get('neg_cap', 0.95)),
                topk=int(params.get('topk', 5)),
                consensus_k=int(params.get('consensus_k', 0)),
                consensus_thr=float(params.get('consensus_thr', 0.45)),
                adaptive_ratio=float(params.get('adaptive_ratio', 0.85)),
                adaptive_diff_floor=float(params.get('adaptive_diff_floor', 0.04)),
                adaptive_trigger_pos_range=float(params.get('adaptive_trigger_pos_range', 1.0)),  # отключено
                adaptive_trigger_neg_range=float(params.get('adaptive_trigger_neg_range', 1.0)),  # отключено
                margin=float(params.get('score_margin', 0.0)),
                ratio=float(params.get('score_ratio', 1.0)),
                confidence=float(params.get('score_confidence', 0.60)),
                allow_unknown=bool(params.get('allow_unknown', True)),
                verbose=bool(params.get('verbose', True)),
            )
        # если явно не задано, и backbone dinov3 — зафиксируем более адекватные пороги
        if str(getattr(detector, 'backbone', '')).startswith('dinov3'):
            if 'min_pos_score' not in (params or {}):
                self.min_pos_score = 0.70         # порог на p (косинус)
            if 'decision_threshold' not in (params or {}):
                self.decision_threshold = 0.15    # p - n разница в косинусе
            if 'consensus_thr' not in (params or {}):
                self.consensus_thr = 0.65         # косинусный порог для консенсуса
        
        # шорткаты
        for k, v in self.config.__dict__.items():
            setattr(self, k, v)

    # агрегирование, как раньше — но без квантильных «порогов»
    def _aggregate_positive(self, sims_pos: np.ndarray) -> np.ndarray:
        """
        Оставляем косинус в [-1..1]. Никаких (x+1)/2!
        """
        if sims_pos.size == 0 or sims_pos.shape[1] == 0:
            return np.zeros((sims_pos.shape[0] if sims_pos.ndim > 0 else 0,), dtype=np.float32)

        maxv = sims_pos.max(axis=1)
        k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
        if k > 1:
            part = np.partition(sims_pos, -k, axis=1)[:, -k:]
            topk_mean = part.mean(axis=1)
        else:
            topk_mean = maxv

        # мягкая агрегация (всё ещё в [-1..1])
        agg = 0.7 * maxv + 0.3 * topk_mean
        return agg.astype(np.float32)

    def _aggregate_negative(self, sims_neg: np.ndarray) -> np.ndarray:
        """
        Негатив — тоже в косинусе [-1..1], берём максимум по колонкам (наиболее похожий негатив).
        """
        if sims_neg.size == 0 or sims_neg.shape[1] == 0:
            return np.zeros((sims_neg.shape[0] if sims_neg.ndim > 0 else 0,), dtype=np.float32)
        neg_sim = np.max(sims_neg, axis=1)
        return neg_sim.astype(np.float32)

    def _consensus_count(self, sims_pos_cls: np.ndarray) -> np.ndarray:
        """
        Считаем количество положительных примеров, у которых косинус >= thr_cos (порог в косинусе!).
        """
        if sims_pos_cls.size == 0:
            return np.zeros((0,), dtype=np.int32)
        return (sims_pos_cls >= float(self.consensus_thr)).sum(axis=1).astype(np.int32)

    # ======= Мультикласс =======
    def score_multiclass(self, mask_vecs: np.ndarray, class_pos: Dict[str, np.ndarray], q_neg: np.ndarray) -> List[Dict[str, Any]]:
        mask_vecs = _to_matrix(mask_vecs)
        D = mask_vecs.shape[1] if mask_vecs.size else 0
        class_pos = {cls: _to_matrix(Q, D) for cls, Q in (class_pos or {}).items()}
        q_neg = _to_matrix(q_neg, D)

        N = mask_vecs.shape[0]
        if N == 0:
            return []

        # neg
        if q_neg.shape[0] == 0:
            neg_scores = np.zeros(N, dtype=np.float32)
        else:
            sims_neg = _cosine_matrix(mask_vecs, q_neg)
            neg_scores = self._aggregate_negative(sims_neg)
        if self.neg_cap is not None:
            neg_scores = np.minimum(neg_scores, float(self.neg_cap))

        # pos per class
        class_scores: Dict[str, np.ndarray] = {}
        class_consensus: Dict[str, np.ndarray] = {}
        if self.verbose:
            print("🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")

        for cls, Q in (class_pos or {}).items():
            if Q.shape[0] == 0:
                continue
            sims_pos = _cosine_matrix(mask_vecs, Q)
            pos_scores = self._aggregate_positive(sims_pos)
            class_scores[cls] = pos_scores
            class_consensus[cls] = self._consensus_count(sims_pos)
            if self.verbose:
                print(f"   Класс '{cls}': скоры={[f'{s:.3f}' for s in pos_scores]}")

        if not class_scores:
            return [
                {'accepted': False, 'class': None, 'pos_score': 0.0, 'neg_score': float(neg_scores[i]), 'confidence': 0.0}
                for i in range(N)
            ]

        classes = list(class_scores.keys())
        pos_matrix = np.stack([class_scores[c] for c in classes], axis=1)  # (N, C)
        best_idx = np.argmax(pos_matrix, axis=1)
        best_pos = pos_matrix[np.arange(N), best_idx]
        second_best = np.partition(pos_matrix, -2, axis=1)[:, -2] if pos_matrix.shape[1] > 1 else np.zeros(N, dtype=np.float32)
        best_cls = [classes[j] for j in best_idx]

        if self.verbose:
            print(f"📊 Скоры мультикласса: pos_avg={np.mean(best_pos):.3f}, neg_avg={np.mean(neg_scores):.3f}")

        decisions: List[Dict[str, Any]] = []
        acc = 0
        eps = 1e-8
        for i in range(N):
            p = float(best_pos[i])
            n = float(neg_scores[i])
            diff = p - n
            ratio_ok = (p / (n + eps)) >= float(self.ratio)

            base_ok = (p >= self.min_pos_score) and (p > n)
            class_sep_ok = (p - float(second_best[i])) >= float(self.class_separation) if len(classes) > 1 else True
            diff_ok = (diff >= self.decision_threshold) or ratio_ok
            consensus_ok = (int(class_consensus.get(best_cls[i], np.zeros(N, dtype=np.int32))[i]) >= int(self.consensus_k))

            ok = base_ok and diff_ok and class_sep_ok and (n <= self.neg_cap) and consensus_ok

            if self.verbose:
                print(f"   Маска {i}: класс={best_cls[i]}, pos={p:.3f}, neg={n:.3f}, diff={diff:+.3f}, "
                      f"sep={(p-second_best[i]):+.3f}, cons={int(consensus_ok)} → принято={ok}")

            decisions.append({
                'accepted': bool(ok),
                'class': best_cls[i] if ok or self.allow_unknown else None,
                'pos_score': p,
                'neg_score': n,
                'confidence': float(np.clip(p, 0.0, 1.0)),
            })
            acc += int(ok)

        if self.verbose:
            print(f"🎯 Принято {acc} из {N} масок")
        return decisions

    # ======= Бинарный режим =======
    def score_and_decide(self, mask_vecs: np.ndarray, q_pos: np.ndarray, q_neg: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
        mask_vecs = _to_matrix(mask_vecs)
        q_pos = _to_matrix(q_pos, mask_vecs.shape[1])
        q_neg = _to_matrix(q_neg, mask_vecs.shape[1])

        sims_pos = _cosine_matrix(mask_vecs, q_pos)
        pos_scores = self._aggregate_positive(sims_pos)

        if q_neg.shape[0] > 0:
            sims_neg = _cosine_matrix(mask_vecs, q_neg)
            neg_scores = self._aggregate_negative(sims_neg)
        else:
            neg_scores = np.zeros(mask_vecs.shape[0], dtype=np.float32)

        accepted = []
        eps = 1e-8
        for i, (p, n) in enumerate(zip(pos_scores, neg_scores)):
            diff = p - n
            ratio_ok = (p / (n + eps)) >= float(self.ratio)
            if (p >= self.min_pos_score) and (diff >= self.decision_threshold or ratio_ok) and (n <= self.neg_cap):
                accepted.append(i)
        return accepted, pos_scores, neg_scores
