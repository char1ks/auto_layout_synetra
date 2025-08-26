#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from ..utils.config import ScoringConfig


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
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    elif A.ndim > 2:
        A = A.reshape(A.shape[0], -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    elif B.ndim > 2:
        B = B.reshape(B.shape[0], -1)

    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)

    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dim mismatch in _cosine_matrix: A{A.shape} vs B{B.shape}")

    A64 = A.astype(np.float64, copy=False)
    B64 = B.astype(np.float64, copy=False)
    A_norm = A64 / (np.linalg.norm(A64, axis=1, keepdims=True) + 1e-12)
    B_norm = B64 / (np.linalg.norm(B64, axis=1, keepdims=True) + 1e-12)

    sims = A_norm @ B_norm.T
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
    sims = np.clip(sims, -1.0, 1.0).astype(np.float32)

    if sims.size and (np.mean(sims >= 0.9995) > 0.9):
        print("   ⚠️  Предупреждение: почти все косинусы >= 0.9995. "
              "Проверьте, не совпадают ли векторы и применяется ли бинарная маска в ROI.")
    return sims


class ScoreCalculator:

    def __init__(self, detector, params: Dict[str, Any] | None = None, config: ScoringConfig | None = None):
        self.detector = detector
        
        # Приоритет: config > params > defaults
        if config is not None:
            self.config = config
        else:
            # Создаем конфигурацию из params для обратной совместимости
            if params is None:
                params = {}
            self.config = ScoringConfig(
                min_pos_score=float(params.get('min_pos_score', 0.62)),
                decision_threshold=float(params.get('decision_threshold', 0.06)),
                class_separation=float(params.get('class_separation', 0.04)),
                neg_cap=float(params.get('neg_cap', 0.90)),
                topk=int(params.get('topk', 5)),
                consensus_k=int(params.get('consensus_k', 0)),
                consensus_thr=float(params.get('consensus_thr', 0.45)),
                adaptive_ratio=float(params.get('adaptive_ratio', 0.85)),
                adaptive_diff_floor=float(params.get('adaptive_diff_floor', 0.04)),
                adaptive_trigger_pos_range=float(params.get('adaptive_trigger_pos_range', 0.20)),
                adaptive_trigger_neg_range=float(params.get('adaptive_trigger_neg_range', 0.20)),
                margin=float(params.get('score_margin', -0.10)),
                ratio=float(params.get('score_ratio', 1.01)),
                confidence=float(params.get('score_confidence', 0.60)),
                allow_unknown=bool(params.get('allow_unknown', True)),
                verbose=bool(params.get('verbose', True))
            )
        
        # Создаем свойства для обратной совместимости
        self.min_pos_score = self.config.min_pos_score
        self.decision_threshold = self.config.decision_threshold
        self.class_separation = self.config.class_separation
        self.neg_cap = self.config.neg_cap
        self.topk = self.config.topk
        self.consensus_k = self.config.consensus_k
        self.consensus_thr = self.config.consensus_thr
        self.adaptive_ratio = self.config.adaptive_ratio
        self.adaptive_diff_floor = self.config.adaptive_diff_floor
        self.adaptive_trigger_pos_range = self.config.adaptive_trigger_pos_range
        self.adaptive_trigger_neg_range = self.config.adaptive_trigger_neg_range
        self.margin = self.config.margin
        self.ratio = self.config.ratio
        self.confidence = self.config.confidence
        self.allow_unknown = self.config.allow_unknown
        self.verbose = self.config.verbose

    def _aggregate_positive(self, sims_pos: np.ndarray) -> np.ndarray:
        if sims_pos.size == 0 or sims_pos.shape[1] == 0:
            return np.zeros((sims_pos.shape[0] if sims_pos.ndim > 0 else 0,), dtype=np.float32)

        maxv = sims_pos.max(axis=1)
        k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
        if k > 1:
            part = np.partition(sims_pos, -k, axis=1)[:, -k:]
            topk_mean = part.mean(axis=1)
        else:
            topk_mean = maxv

        agg = 0.7 * maxv + 0.3 * topk_mean
        pos01 = np.clip((agg + 1.0) * 0.5, 0.0, 1.0)
        return pos01.astype(np.float32)

    def _aggregate_negative(self, sims_neg: np.ndarray) -> np.ndarray:
        if sims_neg.size == 0 or sims_neg.shape[1] == 0:
            return np.zeros((sims_neg.shape[0] if sims_neg.ndim > 0 else 0,), dtype=np.float32)
        neg_sim = np.max(sims_neg, axis=1)
        neg01 = np.clip((neg_sim + 1.0) * 0.5, 0.0, 1.0)
        return neg01.astype(np.float32)

    def _consensus_count(self, sims_pos_cls: np.ndarray) -> np.ndarray:
        if sims_pos_cls.size == 0:
            return np.zeros((0,), dtype=np.int32)
        sims01 = np.clip((sims_pos_cls + 1.0) * 0.5, 0.0, 1.0)
        return (sims01 >= float(self.consensus_thr)).sum(axis=1).astype(np.int32)

    def score_multiclass(
        self,
        mask_vecs: np.ndarray,
        class_pos: Dict[str, np.ndarray],
        q_neg: np.ndarray
    ) -> List[Dict[str, Any]]:
        mask_vecs = _to_matrix(mask_vecs)
        D = mask_vecs.shape[1] if mask_vecs.size else 0
        class_pos = {cls: _to_matrix(Q, D) for cls, Q in (class_pos or {}).items()}
        q_neg = _to_matrix(q_neg, D)

        N = mask_vecs.shape[0]
        if N == 0:
            return []

        if q_neg.shape[0] == 0:
            sims_neg = np.zeros((N, 0), dtype=np.float32)
            neg_scores = np.zeros(N, dtype=np.float32)
        else:
            sims_neg = _cosine_matrix(mask_vecs, q_neg)
            neg_scores = self._aggregate_negative(sims_neg)
        if self.neg_cap is not None:
            neg_scores = np.minimum(neg_scores, float(self.neg_cap))

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
            if self.verbose:
                print("   ❌ Нет валидных классов с эмбеддингами - возвращаем пустой результат")
            return [
                {'accepted': False, 'class': None, 'pos_score': 0.0, 'neg_score': float(neg_scores[i]), 'confidence': 0.0}
                for i in range(N)
            ]

        classes = list(class_scores.keys())
        pos_matrix = np.stack([class_scores[c] for c in classes], axis=1)
        best_idx = np.argmax(pos_matrix, axis=1)
        best_pos = pos_matrix[np.arange(N), best_idx]

        pos_matrix_sorted = np.sort(pos_matrix, axis=1)[:, ::-1]
        second_best = pos_matrix_sorted[:, 1] if pos_matrix_sorted.shape[1] > 1 else np.zeros(N, dtype=np.float32)

        best_cls = [classes[j] for j in best_idx]

        if self.verbose:
            for i in range(N):
                row = ", ".join([f"{c}={class_scores[c][i]:.3f}" for c in classes])
                print(f"     Маска {i}: [{row}] -> выбран {best_cls[i]} ({best_pos[i]:.3f})")

        if self.verbose:
            print(f"📊 Скоры мультикласса: pos_avg={np.mean(best_pos):.3f}, neg_avg={np.mean(neg_scores):.3f}")
            print(f"📊 Диапазоны: pos=[{np.min(best_pos):.3f}, {np.max(best_pos):.3f}], "
                  f"neg=[{np.min(neg_scores):.3f}, {np.max(neg_scores):.3f}]")

        pos_range = float(np.max(best_pos) - np.min(best_pos)) if best_pos.size else 0.0
        neg_range = float(np.max(neg_scores) - np.min(neg_scores)) if neg_scores.size else 0.0
        adaptive_mode = (pos_range < self.adaptive_trigger_pos_range) and (neg_range < self.adaptive_trigger_neg_range)

        if self.verbose:
            print(f"🎯 Пороги: min_pos={self.min_pos_score:.3f}, "
                  f"threshold(diff)={self.decision_threshold:.3f}, "
                  f"class_sep={self.class_separation:.3f}, adaptive_mode={adaptive_mode}")

        class_max = {c: float(np.max(class_scores[c])) for c in classes}
        class_adaptive_thr = {c: (class_max[c] * self.adaptive_ratio) for c in classes}

        eps = 1e-8
        decisions: List[Dict[str, Any]] = []
        accepted_count = 0

        for i in range(N):
            cls_i = best_cls[i]
            pos_s = float(best_pos[i])
            neg_s = float(neg_scores[i]) if i < len(neg_scores) else 0.0
            diff = pos_s - neg_s
            ratio_ok = (pos_s / (neg_s + eps)) >= float(self.ratio)

            cons = int(class_consensus.get(cls_i, np.zeros(N, dtype=np.int32))[i])
            consensus_ok = cons >= int(self.consensus_k)

            base_ok = (pos_s >= self.min_pos_score) and (pos_s > neg_s)

            class_sep = float(pos_s - float(second_best[i]))
            class_sep_ok = (class_sep >= self.class_separation) if len(classes) > 1 else True

            diff_ok = (diff >= self.decision_threshold) or ratio_ok

            accepted_std = base_ok and diff_ok and class_sep_ok and (neg_s <= self.neg_cap) and consensus_ok

            if adaptive_mode:
                thr = float(class_adaptive_thr.get(cls_i, np.max(best_pos)))
                accepted_adapt = base_ok and ( (pos_s >= thr) or (diff >= self.adaptive_diff_floor) ) \
                                 and class_sep_ok and (neg_s <= self.neg_cap) and consensus_ok
                accepted = accepted_std or accepted_adapt
            else:
                accepted = accepted_std

            if self.verbose:
                cons_info = f"cons={cons}/{self.consensus_k}"
                if adaptive_mode:
                    print(f"   Маска {i}: класс={cls_i}, pos={pos_s:.3f}, neg={neg_s:.3f}, "
                          f"diff={diff:+.3f}, sep={class_sep:+.3f}, {cons_info}, "
                          f"adapt_thr={class_adaptive_thr.get(cls_i, 0):.3f}, принято={accepted}")
                else:
                    print(f"   Маска {i}: класс={cls_i}, pos={pos_s:.3f}, neg={neg_s:.3f}, "
                          f"diff={diff:+.3f}, sep={class_sep:+.3f}, {cons_info}, принято={accepted}")

            decisions.append({
                'accepted': bool(accepted),
                'class': cls_i if accepted else (None if not self.allow_unknown else cls_i),
                'pos_score': pos_s,
                'neg_score': neg_s,
                'confidence': float(np.clip(pos_s, 0.0, 1.0)),
            })
            accepted_count += int(bool(accepted))

        if self.verbose:
            print(f"🎯 Принято {accepted_count} из {N} масок")
        return decisions

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
        for i, (p, n) in enumerate(zip(pos_scores, neg_scores)):
            diff = p - n
            ratio_ok = (p / (n + 1e-8)) >= float(self.ratio)
            if (p >= self.min_pos_score) and (diff >= self.decision_threshold or ratio_ok) and (n <= self.neg_cap):
                accepted.append(i)

        return accepted, pos_scores, neg_scores
