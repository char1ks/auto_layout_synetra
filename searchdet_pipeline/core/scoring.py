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
        accepted_indices = []
        for i in range(len(pos_scores)):
            pos = pos_scores[i]
            neg = neg_scores[i]
            
            # Правила принятия решений
            if (pos > neg + self.margin and
                neg < self.neg_cap and
                pos / (neg + 1e-8) > self.ratio and
                pos > self.confidence):
                accepted_indices.append(i)
        
        print(f"📊 Скоры: pos_avg={np.mean(pos_scores):.3f}, neg_avg={np.mean(neg_scores):.3f}")
        print(f"🎯 Принято {len(accepted_indices)} из {len(pos_scores)} масок")
        return accepted_indices
    
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
        
        # Берем среднее арифметическое и переводим в [0,1]
        pos_sim = np.mean(sims_pos, axis=1)  # [-1,1]
        pos_score = np.clip((pos_sim + 1) / 2, 0, 1)  # [0,1]
        
        return pos_score
    
    def _aggregate_negative(self, sims_neg):
        """Агрегирует negative similarities в единую оценку."""
        if sims_neg.shape[1] == 0:
            return np.zeros(sims_neg.shape[0])
        
        # Берем максимальное сходство и переводим в [0,1]
        neg_sim = np.max(sims_neg, axis=1)  # [-1,1]
        neg_score = np.clip((neg_sim + 1) / 2, 0, 1)  # [0,1]
        
        return neg_score
