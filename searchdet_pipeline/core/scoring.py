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

        # --- негативы: сырые косинусы ---
        if q_neg.shape[0] == 0:
            sims_neg = np.zeros((N, 0), dtype=np.float32)
            neg_raw = np.zeros(N, dtype=np.float32)
        else:
            sims_neg = _cosine_matrix(mask_vecs, q_neg)
            neg_raw = self._aggregate_negative(sims_neg)

        # для принятия решений используем clamp до [0, neg_cap]
        neg_for_decision = np.clip(np.maximum(neg_raw, 0.0), 0.0,
                                   (self.neg_cap if self.neg_cap is not None else 1.0)).astype(np.float32)

        if self.verbose:
            print("🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")
            print("   ℹ️ |mask_vec| mean≈", float(np.mean(np.linalg.norm(mask_vecs, axis=1))))
            print("   ℹ️ |Q_pos| mean≈", float(np.mean([np.mean(np.linalg.norm(Q, axis=1)) for Q in class_pos.values() if len(Q)>0] or [0])))

        # --- по классам: сырые pos косинусы ---
        class_scores: Dict[str, np.ndarray] = {}
        class_consensus: Dict[str, np.ndarray] = {}

        for cls, Q in (class_pos or {}).items():
            if Q.shape[0] == 0:
                if self.verbose:
                    print(f"   📊 Класс '{cls}': 0 примеров")
                continue
            sims_pos = _cosine_matrix(mask_vecs, Q)      # [-1,1]
            pos_scores = self._aggregate_positive(sims_pos)
            class_scores[cls] = pos_scores
            class_consensus[cls] = self._consensus_count(sims_pos)
            if self.verbose:
                s_list = [f"{s:.3f}" for s in pos_scores]
                print(f"   📊 Класс '{cls}': {Q.shape[0]} примеров → скоры={s_list}")

        if not class_scores:
            if self.verbose:
                print("   ❌ Нет валидных классов с эмбеддингами - возвращаем пустой результат")
            return [
                {'accepted': False, 'class': None, 'pos_score': 0.0,
                 'neg_score': float(neg_for_decision[i]), 'confidence': 0.0}
                for i in range(N)
            ]

        classes = list(class_scores.keys())
        pos_matrix = np.stack([class_scores[c] for c in classes], axis=1)  # [N, C]
        best_idx = np.argmax(pos_matrix, axis=1)
        best_pos = pos_matrix[np.arange(N), best_idx]                      # сырые косинусы
        second_best = np.partition(pos_matrix, -2, axis=1)[:, -2] if pos_matrix.shape[1] > 1 \
                      else best_pos - 0.0  # так sep≈0 при одном классе
        best_cls = [classes[j] for j in best_idx]

        if self.verbose:
            for i in range(N):
                row = ", ".join([f"{c}={class_scores[c][i]:.3f}" for c in classes])
                print(f"     Маска {i}: [{row}] → выбран {best_cls[i]} ({best_pos[i]:.3f})")

        # сводные метрики
        pos_min, pos_max = float(np.min(best_pos)), float(np.max(best_pos))
        neg_avg = float(np.mean(neg_raw)) if neg_raw.size else 0.0
        if self.verbose:
            print(f"📊 Скоры мультикласса: pos_avg={np.mean(best_pos):.3f}, neg_avg={neg_avg:.3f}")

        pos_range = pos_max - pos_min
        neg_range = float(np.max(neg_raw) - np.min(neg_raw)) if neg_raw.size else 0.0

        # Аккуратная анти-коллапс отсечка:
        # включаем ТОЛЬКО если нет негативов, и все pos одинаково «запредельно» высокие
        use_quantile_gate = (q_neg.shape[0] == 0) and (pos_max > 0.95) and (pos_range < 0.02)

        if self.verbose:
            print(f"🎯 Пороги: min_pos={self.min_pos_score:.3f}, diff_thr={self.decision_threshold:.3f}, "
                  f"class_sep={self.class_separation:.3f}, quantile_gate={use_quantile_gate}")

        q85 = float(np.quantile(best_pos, 0.85)) if use_quantile_gate else None
        if use_quantile_gate and self.verbose:
            print(f"   📐 Квантильная отсечка: Q85={q85:.3f}")

        eps = 1e-8
        out: List[Dict[str, Any]] = []
        accepted_count = 0

        for i in range(N):
            cls_i = best_cls[i]
            pos_s = float(best_pos[i])                 # косинус [-1,1]
            neg_raw_i = float(neg_raw[i]) if i < len(neg_raw) else 0.0
            neg_i = float(neg_for_decision[i]) if i < len(neg_for_decision) else 0.0

            # базовые условия
            base_ok = (pos_s >= self.min_pos_score) and (pos_s > neg_i)
            class_sep = float(pos_s - float(second_best[i]))
            class_sep_ok = (class_sep >= self.class_separation) if len(classes) > 1 else True

            diff = pos_s - neg_i - self.margin
            ratio_ok = (pos_s / (neg_i + eps)) >= float(self.ratio) if neg_i > 0 else (pos_s >= self.min_pos_score)
            diff_ok = (diff >= self.decision_threshold) or ratio_ok

            cons = int(class_consensus.get(cls_i, np.zeros(N, dtype=np.int32))[i])
            consensus_ok = cons >= int(self.consensus_k)

            # квантильная отсечка (только при спец-условиях выше)
            quant_ok = True
            if use_quantile_gate:
                quant_ok = (pos_s >= q85)

            accepted = bool(base_ok and diff_ok and class_sep_ok and consensus_ok and quant_ok)

            if self.verbose:
                cons_info = f"cons={cons}/{self.consensus_k}"
                extra = f", neg_raw={neg_raw_i:.3f}, neg={neg_i:.3f}, diff={diff:+.3f}, sep={class_sep:+.3f}"
                if use_quantile_gate:
                    extra += f", q85={q85:.3f}"
                print(f"   Маска {i}: класс={cls_i}, pos={pos_s:.3f}{extra}, {cons_info} → принято={accepted}")

            out.append({
                'accepted': accepted,
                'class': cls_i if (accepted or self.allow_unknown) else None,
                'pos_score': pos_s,                  # СЫРОЙ КОСИНУС
                'neg_score': neg_i,                  # уже с clamp [0, neg_cap]
                'neg_score_raw': neg_raw_i,          # для отладки
                'confidence': float(np.clip(0.5 * (pos_s + 1.0), 0.0, 1.0)),  # в [0,1] только для UI
            })
            if accepted:
                accepted_count += 1

        if self.verbose:
            print(f"🎯 Принято {accepted_count} из {N} масок")
        return out

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
