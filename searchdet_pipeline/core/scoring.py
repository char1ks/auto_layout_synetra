#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скоринг масок и принятие решений.
"""

import numpy as np


class ScoreCalculator:
    """Вычисление скоров и принятие решений."""
    
    def __init__(self, detector, params=None):
        if params is None:
            params = {}
            
        self.detector = detector
        
        # Загружаем параметры скоринга
        self.margin = params.get('score_margin', -0.10)
        self.ratio = params.get('score_ratio', 0.80)
        self.confidence = params.get('score_confidence', 0.60)
        self.neg_cap = params.get('neg_cap', 0.95)
        self.topk = params.get('topk', 3)
        # Консенсус
        self.consensus_k = params.get('consensus_k', 3)
        self.consensus_thr = params.get('consensus_thr', 0.60)
    
    def score_and_decide(self, mask_vecs, q_pos, q_neg):
        """Вычисляет скоры и принимает решения."""
        if mask_vecs.shape[0] == 0:
            return []
        
        # 1. Вычисляем матрицы сходства
        sims_pos = self._cosine_matrix(mask_vecs, q_pos)  # [-1..1]
        pos_scores = self._aggregate_positive(sims_pos)    # [0..1]
        
        # 2. Вычисляем similarity с negative
        if q_neg.shape[0] > 0:
            sims_neg = self._cosine_matrix(mask_vecs, q_neg)
            neg_scores = np.max(sims_neg, axis=1)  # берем максимальное сходство с negative
            neg_scores = np.clip((neg_scores + 1) / 2, 0, 1)  # [-1,1] → [0,1]
        else:
            neg_scores = np.zeros(mask_vecs.shape[0])
        
        # 3. Принимаем решения
        # Консенсус по позитивам: считаем количество попаданий >= consensus_thr
        consensus_cnt = np.zeros_like(pos_scores)
        if sims_pos.size > 0 and q_pos.shape[0] > 0:
            sims_pos_01 = (sims_pos + 1.0) * 0.5  # [-1,1] -> [0,1]
            consensus_cnt = (sims_pos_01 >= self.consensus_thr).sum(axis=1)

        accepted_indices = []
        for i in range(len(pos_scores)):
            pos = pos_scores[i]
            neg = neg_scores[i]
            cons = consensus_cnt[i] if hasattr(consensus_cnt, '__len__') else 0
            
            # Более гибкая логика принятия решений
            score_diff = pos - neg  # положительное значение = хорошо
            score_balance = pos / (neg + 1e-8)  # больше 1.0 = хорошо
            
            # Основные критерии (любой из них может сработать)
            criterion1 = pos >= self.confidence and score_diff >= self.margin  # классический
            criterion2 = pos >= (self.confidence * 0.8) and score_balance >= self.ratio  # баланс важнее
            criterion3 = score_diff >= (self.margin + 0.1) and pos >= (self.confidence * 0.7)  # большой отрыв
            
            # Дополнительные проверки
            neg_check = neg <= self.neg_cap
            consensus_check = cons >= max(1, self.topk if self.consensus_k is None else self.consensus_k)
            
            if (criterion1 or criterion2 or criterion3) and neg_check and consensus_check:
                accepted_indices.append(i)
        
        print(f"📊 Скоры: pos_avg={np.mean(pos_scores):.3f}, neg_avg={np.mean(neg_scores):.3f}")
        print(f"📊 Диапазоны: pos=[{np.min(pos_scores):.3f}, {np.max(pos_scores):.3f}], neg=[{np.min(neg_scores):.3f}, {np.max(neg_scores):.3f}]")
        print(f"🎯 Принято {len(accepted_indices)} из {len(pos_scores)} масок")
        
        # Отладочная информация для отклоненных масок
        if len(accepted_indices) < len(pos_scores):
            rejected_count = len(pos_scores) - len(accepted_indices)
            print(f"❌ Отклонено {rejected_count} масок:")
            for i in range(min(3, len(pos_scores))):  # показываем первые 3
                if i not in accepted_indices:
                    pos, neg = pos_scores[i], neg_scores[i]
                    diff = pos - neg
                    ratio = pos / (neg + 1e-8)
                    print(f"   Маска {i}: pos={pos:.3f}, neg={neg:.3f}, diff={diff:+.3f}, ratio={ratio:.2f}")
            if rejected_count > 3:
                print(f"   ... и еще {rejected_count - 3} масок")
        return accepted_indices, pos_scores, neg_scores
    
    def _cosine_matrix(self, A, B):
        """Вычисляет матрицу косинусных расстояний между A и B."""
        if A.shape[0] == 0 or B.shape[0] == 0:
            return np.array([]).reshape(A.shape[0], B.shape[0])
        
        # Нормализуем векторы
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        
        # Косинусное сходство = скалярное произведение нормализованных векторов
        return np.dot(A_norm, B_norm.T)
    
    def _aggregate_positive(self, sims_pos):
        """Агрегирует positive similarities в единую оценку."""
        if sims_pos.shape[1] == 0:
            return np.zeros(sims_pos.shape[0])
        # Гибридная агрегция: 0.7*max + 0.3*topk_mean по косинусным сходствам
        maxv = sims_pos.max(axis=1)  # [-1,1]
        k = min(self.topk, sims_pos.shape[1]) if sims_pos.shape[1] > 0 else 1
        if k > 1:
            # top-k среднее по каждой строке
            part = np.partition(sims_pos, -k, axis=1)[:, -k:]
            topk_mean = part.mean(axis=1)  # [-1,1]
        else:
            topk_mean = maxv
        agg = 0.7 * maxv + 0.3 * topk_mean  # [-1,1]
        pos_score = np.clip((agg + 1.0) / 2.0, 0.0, 1.0)  # [0,1]
        return pos_score
    
    def _aggregate_negative(self, sims_neg):
        """Агрегирует negative similarities в единую оценку."""
        if sims_neg.shape[1] == 0:
            return np.zeros(sims_neg.shape[0])
        
        # Берем максимальное сходство и переводим в [0,1]
        neg_sim = np.max(sims_neg, axis=1)  # [-1,1]
        neg_score = np.clip((neg_sim + 1) / 2, 0, 1)  # [0,1]
        
        return neg_score


    def score_multiclass(self, mask_vecs, class_pos, q_neg):
        """Скоринг для мультикласса.
        
        Args:
            mask_vecs: np.ndarray [N, D] — эмбеддинги масок
            class_pos: dict[str, np.ndarray] — по каждому классу позитивы [M_cls, D]
            q_neg: np.ndarray [K, D] — негативы
        
        Returns:
            decisions: list[dict] длины N, у каждого:
                {'accepted': bool, 'class': str|None, 'pos_score': float, 'neg_score': float, 'confidence': float}
        """
        N = mask_vecs.shape[0]
        if N == 0:
            return []
        
        # Предрасчёт негативов
        if q_neg is None or q_neg.shape[0] == 0:
            sims_neg = np.zeros((N, 0), dtype=np.float32)
            neg_scores = np.zeros(N, dtype=np.float32)
        else:
            sims_neg = mask_vecs @ q_neg.T  # косинус если вектора уже L2-нормированы
            neg_scores = self._aggregate_negative(sims_neg)
        
        # По классам
        best_cls = [None]*N
        best_pos = np.zeros(N, dtype=np.float32)
        for cls, q_pos in (class_pos or {}).items():
            if q_pos is None or q_pos.shape[0] == 0:
                continue
            sims_pos = mask_vecs @ q_pos.T
            pos_scores = self._aggregate_positive(sims_pos)
            # Обновляем лучший класс по pos_score
            better = pos_scores > best_pos
            best_pos = np.where(better, pos_scores, best_pos)
            for i in range(N):
                if better[i]:
                    best_cls[i] = cls
        
        # Итоговые решения
        decisions = []
        threshold = float(self.params.get('decision_threshold', 0.15))  # gap между pos и neg
        min_pos = float(self.params.get('min_pos_score', 0.55))
        for i in range(N):
            pos_s = float(best_pos[i])
            neg_s = float(neg_scores[i])
            conf = pos_s - neg_s
            accepted = (pos_s >= min_pos) and (conf >= threshold) and (best_cls[i] is not None)
            decisions.append({
                'accepted': bool(accepted),
                'class': best_cls[i],
                'pos_score': pos_s,
                'neg_score': neg_s,
                'confidence': float(np.clip(pos_s, 0.0, 1.0))  # совместимо с текущим UI
            })
        return decisions
