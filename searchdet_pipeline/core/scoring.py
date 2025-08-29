#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from ..utils.config import ScoringConfig


# --------------- утилиты ---------------

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
    """
    Косинусы в сыром виде в диапазоне [-1, 1].
    Ожидается, что входы уже L2-нормированы (но мы страхуемся).
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    if A.ndim == 1: A = A.reshape(1, -1)
    elif A.ndim > 2: A = A.reshape(A.shape[0], -1)
    if B.ndim == 1: B = B.reshape(1, -1)
    elif B.ndim > 2: B = B.reshape(B.shape[0], -1)

    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dim mismatch in _cosine_matrix: A{A.shape} vs B{B.shape}")

    # на всякий случай нормализуем (если уже L2 - это дёшево)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

    sims = A @ B.T
    sims = np.clip(sims, -1.0, 1.0).astype(np.float32)
    return sims


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N,D), b: (M,D) -> (N,M)
    return a @ b.T

def _agg_scores(sims: np.ndarray, mode: str = "mean_topk", topk: int = 2) -> np.ndarray:
    """
    sims: (N, K_i) — похожесть масок к K_i примерам одного класса.
    Возвращает (N,) — агрегированный pos по классу.
    """
    if sims.size == 0:
        return np.full((sims.shape[0],), -1.0, dtype=np.float32)
    if mode == "max":
        return sims.max(axis=1)
    if mode == "mean":
        return sims.mean(axis=1)
    # mean_topk
    k = min(topk, sims.shape[1])
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


# --------------- скорер ---------------

class ScoreCalculator:
    """
    Главные отличия от старой логики:
    1) работаем с сырыми косинусами [-1,1], не маппим в [0,1] до решения;
    2) для решения используем neg_for_decision=max(neg_raw,0) — отрицательные
       косинусы негативов не улучшают diff, а просто нулируются;
    3) консенсус-порог, переданный как 0..1 (как раньше), переводим во внутренний
       косинусный порог: thr_cos = 2*thr01 - 1.
    """

    def __init__(self, detector, params: Dict[str, Any] | None = None, config: ScoringConfig | None = None):
        self.detector = detector

        # приоритет: config > params > defaults (совместимость с вашим CLI)
        if config is not None:
            self.config = config
        else:
            if params is None:
                params = {}
            self.config = ScoringConfig(
                min_pos_score=float(params.get('min_pos_score', 0.62)),      # в КОСИНУСАХ
                decision_threshold=float(params.get('decision_threshold', 0.06)),  # тоже в косинусах
                class_separation=float(params.get('class_separation', 0.04)),     # разница с 2-м лучшим
                neg_cap=float(params.get('neg_cap', 0.90)),                  # ограничение сверху для neg_for_decision
                topk=int(params.get('topk', 5)),
                consensus_k=int(params.get('consensus_k', 0)),
                consensus_thr=float(params.get('consensus_thr', 0.45)),       # ПРИНИМАЕМ 0..1, ниже переведём в косинус
                adaptive_ratio=float(params.get('adaptive_ratio', 0.85)),     # оставлено для совместимости (ViT не нужен)
                adaptive_diff_floor=float(params.get('adaptive_diff_floor', 0.04)),
                adaptive_trigger_pos_range=float(params.get('adaptive_trigger_pos_range', 0.20)),
                adaptive_trigger_neg_range=float(params.get('adaptive_trigger_neg_range', 0.20)),
                margin=float(params.get('score_margin', 0.00)),               # дополнительный сдвиг порога diff
                ratio=float(params.get('score_ratio', 1.01)),                 # p/(neg+eps) — оставлено
                confidence=float(params.get('score_confidence', 0.60)),       # min визуальной уверенности (на вывод)
                allow_unknown=bool(params.get('allow_unknown', True)),
                verbose=bool(params.get('verbose', True))
            )

        # alias для краткости
        self.min_pos_score = self.config.min_pos_score
        self.decision_threshold = self.config.decision_threshold
        self.class_separation = self.config.class_separation
        self.neg_cap = self.config.neg_cap
        self.topk = self.config.topk
        self.consensus_k = self.config.consensus_k
        self.consensus_thr_01 = self.config.consensus_thr  # 0..1 на входе
        self.adaptive_ratio = self.config.adaptive_ratio
        self.adaptive_diff_floor = self.config.adaptive_diff_floor
        self.adaptive_trigger_pos_range = self.config.adaptive_trigger_pos_range
        self.adaptive_trigger_neg_range = self.config.adaptive_trigger_neg_range
        self.margin = self.config.margin
        self.ratio = self.config.ratio
        self.confidence_min = self.config.confidence
        self.allow_unknown = self.config.allow_unknown
        self.verbose = self.config.verbose

        # переведём consensus_thr в косинус (-1..1)
        self.consensus_thr_cos = float(2 * self.consensus_thr_01 - 1.0)

        # ▼ Новый параметр: режим агрегации positive-скоров
        #   'max'       — как в тестовом скрипте (рекомендуется)
        #   'mean_topk' — среднее по top-k (k = self.topk)
        #   'mean'      — среднее по всем positive
        self.pos_agg = str((params or {}).get('pos_agg', getattr(detector, 'pos_agg', 'max'))).lower()
        if self.verbose:
            print(f"   ⚙️ POS_AGG_MODE = {self.pos_agg}")

    # ----- агрегаторы

    def _aggregate_positive(self, sims_pos: np.ndarray) -> np.ndarray:
        """
        Агрегируем сырые косинусы [-1,1] по positive-примерам.
        По умолчанию — 'max' (как в тестовом dinov3_similarity.py).
        """
        if sims_pos.size == 0 or sims_pos.shape[1] == 0:
            return np.zeros((sims_pos.shape[0] if sims_pos.ndim > 0 else 0,), dtype=np.float32)

        mode = self.pos_agg
        if mode in ("max", "top1"):
            pos = sims_pos.max(axis=1)
        elif mode in ("mean_topk", "topk"):
            k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
            if k > 1:
                part = np.partition(sims_pos, -k, axis=1)[:, -k:]
                pos = part.mean(axis=1)
            else:
                pos = sims_pos.max(axis=1)
        elif mode == "mean":
            pos = sims_pos.mean(axis=1)
        else:
            # fallback: старая формула (0.7*max + 0.3*mean_topk)
            maxv = sims_pos.max(axis=1)
            k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
            if k > 1:
                part = np.partition(sims_pos, -k, axis=1)[:, -k:]
                topk_mean = part.mean(axis=1)
            else:
                topk_mean = maxv
            pos = 0.7 * maxv + 0.3 * topk_mean

        return np.clip(pos.astype(np.float32), -1.0, 1.0)

    def _aggregate_negative(self, sims_neg: np.ndarray) -> np.ndarray:
        """Берём максимум по негативам (сырые косинусы, [-1,1])."""
        if sims_neg.size == 0 or sims_neg.shape[1] == 0:
            return np.zeros((sims_neg.shape[0] if sims_neg.ndim > 0 else 0,), dtype=np.float32)
        neg_raw = np.max(sims_neg, axis=1)
        neg_raw = np.clip(neg_raw, -1.0, 1.0).astype(np.float32)
        return neg_raw

    def _consensus_count(self, sims_pos_cls: np.ndarray) -> np.ndarray:
        """Сколько positive-примеров у класса дают косинус >= consensus_thr_cos."""
        if sims_pos_cls.size == 0:
            return np.zeros((0,), dtype=np.int32)
        return (sims_pos_cls >= self.consensus_thr_cos).sum(axis=1).astype(np.int32)

    # ----- основной мультикласс-скоринг

    def score_multiclass(
        self,
        mask_vecs: np.ndarray,                    # (N,D), L2-normed
        q_pos: dict[str, np.ndarray],             # class -> (K_i,D)
        q_neg: np.ndarray | None,                 # (K_neg,D) or None/(0,*)
        *,
        online_negatives: np.ndarray | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает decisions: List[dict], где dict = {
            'accepted': bool, 'cls': str|None, 'pos': float, 'neg': float,
            'diff': float, 'ratio': float, 'confidence': float
        }
        """
        pos_agg = self.pos_agg
        topk = self.topk
        min_pos = self.min_pos_score
        diff_thr = self.decision_threshold
        ratio_thr = self.ratio
        top2_margin = self.class_separation
        require_two_pos = False # not in config, use default

        N, D = mask_vecs.shape
        if not q_pos:
            return [{'accepted': False, 'cls': None, 'pos': 0.0, 'neg': 0.0,
                     'diff': 0.0, 'ratio': 0.0, 'confidence': 0.0} for _ in range(N)]

        # Собираем pos-скор по всем классам
        classes = sorted(q_pos.keys())
        pos_all = []
        for cls in classes:
            Q = q_pos[cls]  # (K_i, D)
            sims = _cos(mask_vecs, Q)            # (N, K_i), уже [-1..1]
            pos_cls = _agg_scores(sims, pos_agg, topk)
            if require_two_pos and Q.shape[0] < 2:
                # штрафуем классы с 1 примером, чтобы не «долбить» max по одному
                pos_cls -= 0.05
            pos_all.append(pos_cls)
        pos_all = np.stack(pos_all, 1)            # (N, C)
        best_idx = pos_all.argmax(axis=1)         # (N,)
        best_pos = pos_all[np.arange(N), best_idx]
        # второй лучший класс (для отрыва/top2-margin)
        pos_sorted = np.sort(pos_all, axis=1)
        second_pos = pos_sorted[:, -2] if pos_sorted.shape[1] >= 2 else np.full((N,), -1.0, dtype=np.float32)

        # NEG: реальные + онлайн
        neg_pool = []
        if q_neg is not None and q_neg.size:
            neg_pool.append(q_neg)
        if online_negatives is not None and online_negatives.size:
            neg_pool.append(online_negatives)
        if neg_pool:
            NEG = np.concatenate(neg_pool, axis=0)     # (K_neg, D)
            neg_scores = _cos(mask_vecs, NEG).max(axis=1)  # (N,)
        else:
            # Если нет негативов — используем второй лучший класс как суррогат neg
            neg_scores = second_pos.copy()

        # Гейты
        eps = 1e-3
        ratios = best_pos / np.maximum(neg_scores, 0.01)   # чтобы не делить на 0
        diffs = best_pos - np.maximum(neg_scores, 0.0)
        margins = best_pos - second_pos

        decisions = []
        for i in range(N):
            cls = classes[best_idx[i]]
            pos = float(np.clip(best_pos[i], -1.0, 1.0))
            neg = float(np.clip(neg_scores[i], -1.0, 1.0))
            diff = float(diffs[i])
            ratio = float(ratios[i])
            margin2 = float(margins[i])

            accept = (pos >= min_pos) and (diff >= diff_thr) and (ratio >= ratio_thr) and (margin2 >= top2_margin)
            conf = max(0.0, min(1.0, 0.5 * (pos - min_pos) + 0.5 * (diff - diff_thr)))

            decisions.append({
                'accepted': bool(accept),
                'cls': cls if accept else cls,
                'pos': pos,
                'neg': neg,
                'diff': diff,
                'ratio': ratio,
                'confidence': float(conf),
            })

        return decisions

    # ----- бинарный режим (оставлен для совместимости/APIs)

    def score_and_decide(self, mask_vecs: np.ndarray, q_pos: np.ndarray, q_neg: np.ndarray
                         ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        mask_vecs = _to_matrix(mask_vecs)
        q_pos = _to_matrix(q_pos, mask_vecs.shape[1])
        q_neg = _to_matrix(q_neg, mask_vecs.shape[1])

        sims_pos = _cosine_matrix(mask_vecs, q_pos)  # [-1,1]
        pos_scores = self._aggregate_positive(sims_pos)  # [-1,1]

        if q_neg.shape[0] > 0:
            sims_neg = _cosine_matrix(mask_vecs, q_neg)   # [-1,1]
            neg_raw = self._aggregate_negative(sims_neg)  # [-1,1]
        else:
            neg_raw = np.zeros(mask_vecs.shape[0], dtype=np.float32)

        neg_for_decision = np.clip(np.maximum(neg_raw, 0.0), 0.0,
                                   (self.neg_cap if self.neg_cap is not None else 1.0)).astype(np.float32)

        accepted = []
        eps = 1e-8
        for i, (p, n) in enumerate(zip(pos_scores, neg_for_decision)):
            diff = p - n - self.margin
            ratio_ok = (p / (n + eps)) >= float(self.ratio) if n > 0 else (p >= self.min_pos_score)
            if (p >= self.min_pos_score) and ((diff >= self.decision_threshold) or ratio_ok) and (p > n):
                accepted.append(i)

        return accepted, pos_scores, neg_for_decision
