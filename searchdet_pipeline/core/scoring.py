#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скоринг масок и принятие решений.
"""

import numpy as np


class ScoreCalculator:
    """Вычисление оценок и принятие решений о масках."""
    
    def __init__(self, detector):
        self.detector = detector
    
    def score_and_decide(self, mask_vecs, q_pos, q_neg):
        """Вычисляет скоры и принимает решения о масках."""
        if mask_vecs.shape[0] == 0:
            return []
        
        # 1. Вычисляем similarity с positive
        sims_pos_raw = self._cosine_matrix(mask_vecs, q_pos)  # [-1..1]
        pos_score = self._aggregate_positive(sims_pos_raw)    # [0..1]
        
        # 2. Вычисляем similarity с negative
        if q_neg.shape[0] > 0:
            sims_neg = self._cosine_matrix(mask_vecs, q_neg)
            neg_score = np.max(sims_neg, axis=1)  # берем максимальное сходство с negative
            neg_score = np.clip((neg_score + 1) / 2, 0, 1)  # [-1,1] → [0,1]
        else:
            neg_score = np.zeros(mask_vecs.shape[0])
        
        # 3. Применяем правила принятия решений
        accepted = self._apply_acceptance_rules(pos_score, neg_score)
        
        print(f"📊 Скоры: pos_avg={pos_score.mean():.3f}, neg_avg={neg_score.mean():.3f}")
        print(f"🎯 Принято {len(accepted)} из {len(mask_vecs)} масок")
        
        return accepted
    
    def _cosine_matrix(self, A, B):
        """Вычисляет матрицу косинусных расстояний между A и B."""
        if A.shape[0] == 0 or B.shape[0] == 0:
            return np.array([]).reshape(A.shape[0], B.shape[0])
        
        # Нормализуем векторы
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        
        # Косинусное сходство = скалярное произведение нормализованных векторов
        return np.dot(A_norm, B_norm.T)
    
    def _aggregate_positive(self, sims_pos_raw):
        """Агрегирует positive similarities в единую оценку."""
        if sims_pos_raw.shape[1] == 0:
            return np.zeros(sims_pos_raw.shape[0])
        
        # Берем среднее арифметическое и переводим в [0,1]
        pos_sim = np.mean(sims_pos_raw, axis=1)  # [-1,1]
        pos_score = np.clip((pos_sim + 1) / 2, 0, 1)  # [0,1]
        
        return pos_score
    
    def _apply_acceptance_rules(self, pos_score, neg_score):
        """Применяет правила приёма масок."""
        # Получаем параметры
        min_confidence = getattr(self.detector, 'min_confidence', 0.60)
        margin = getattr(self.detector, 'margin', -0.10)
        ratio = getattr(self.detector, 'ratio', 0.80)
        neg_cap = getattr(self.detector, 'neg_cap', 0.95)
        
        accepted_indices = []
        
        for i in range(len(pos_score)):
            pos = pos_score[i]
            neg = neg_score[i]
            
            # Правило 1: минимальная уверенность
            if pos < min_confidence:
                continue
            
            # Правило 2: margin (pos должен быть выше neg + margin)
            if pos < neg + margin:
                continue
            
            # Правило 3: ratio (pos/neg >= ratio, если neg > 0)
            if neg > 0.1 and pos / neg < ratio:
                continue
            
            # Правило 4: negative cap (neg не должен быть слишком высоким)
            if neg > neg_cap:
                continue
            
            accepted_indices.append(i)
        
        return accepted_indices
