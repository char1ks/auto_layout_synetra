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
        self.params = params 
        # Загружаем параметры скоринга
        self.margin = params.get('score_margin', -0.10)
        self.ratio = params.get('score_ratio', 0.80)
        self.confidence = params.get('score_confidence', 0.60)
        self.neg_cap = params.get('neg_cap', 0.95)
        self.topk = params.get('topk', 3)
        # Консенсус
        self.consensus_k = params.get('consensus_k', 3)
        self.consensus_thr = params.get('consensus_thr', 0.60)

        # новые пороги для open-set
        self.min_pos_score = params.get('min_pos_score', 0.70)          # абсолютный минимум
        self.decision_threshold = params.get('decision_threshold', 0.10) # отрыв от негатива
        self.class_separation = params.get('class_separation', 0.08)     # отрыв от 2-го класса
        self.allow_unknown = params.get('allow_unknown', True)
    
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
            sims_neg = self._cosine_matrix(mask_vecs, q_neg)  # правильное косинусное сходство
            neg_scores = self._aggregate_negative(sims_neg)
        
        # По классам - собираем все скоры сначала
        class_scores = {}  # {class_name: np.array of scores}
        print(f"🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")
        for cls, q_pos in (class_pos or {}).items():
            if q_pos is None or q_pos.shape[0] == 0:
                continue
            sims_pos = self._cosine_matrix(mask_vecs, q_pos)  # правильное косинусное сходство
            pos_scores = self._aggregate_positive(sims_pos)
            class_scores[cls] = pos_scores
            print(f"   Класс '{cls}': скоры={[f'{s:.3f}' for s in pos_scores]}")
        
        # Теперь для каждой маски выбираем класс с максимальным скором
        best_cls = [None]*N
        best_pos = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            best_score = -1.0
            best_class = None
            
            # Находим класс с максимальным скором для маски i
            for cls, scores in class_scores.items():
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_class = cls
            
            best_pos[i] = best_score
            best_cls[i] = best_class
            
            # Отладочная информация с проверкой
            scores_info = ", ".join([f"{cls}={scores[i]:.3f}" for cls, scores in class_scores.items()])
            
            # Дополнительная проверка: найдем реальный максимум
            actual_max_score = max([scores[i] for scores in class_scores.values()])
            actual_max_class = None
            for cls, scores in class_scores.items():
                if scores[i] == actual_max_score:
                    actual_max_class = cls
                    break
            
            print(f"     Маска {i}: [{scores_info}] -> выбран {best_class} ({best_score:.3f})")
            if best_class != actual_max_class:
                print(f"     ⚠️  ОШИБКА: должен быть выбран {actual_max_class} ({actual_max_score:.3f})!")
                # Исправляем
                best_pos[i] = actual_max_score
                best_cls[i] = actual_max_class
        
        # Отладочная информация
        print(f"📊 Скоры мультикласса: pos_avg={np.mean(best_pos):.3f}, neg_avg={np.mean(neg_scores):.3f}")
        print(f"📊 Диапазоны: pos=[{np.min(best_pos):.3f}, {np.max(best_pos):.3f}], neg=[{np.min(neg_scores):.3f}, {np.max(neg_scores):.3f}]")
        
        # Итоговые решения с адаптивными порогами
        decisions = []
        # Понижаем пороги для случаев с низким различием
        min_pos = float(self.params.get('min_pos_score', 0.40))  # понижен с 0.50
        threshold = float(self.params.get('decision_threshold', -0.10))  # понижен для большей гибкости
        
        # Адаптивная логика: если все скоры близки, используем относительное сравнение
        pos_range = np.max(best_pos) - np.min(best_pos)
        neg_range = np.max(neg_scores) - np.min(neg_scores)
        adaptive_mode = pos_range < 0.1 and neg_range < 0.1  # если диапазон мал
        
        print(f"🎯 Пороги: min_pos={min_pos:.3f}, threshold={threshold:.3f}, adaptive_mode={adaptive_mode}")
        
        accepted_count = 0
        for i in range(N):
            pos_s = float(best_pos[i])
            neg_s = float(neg_scores[i])
            conf = pos_s - neg_s
            
            if adaptive_mode:
                # В адаптивном режиме принимаем лучшие маски относительно других
                pos_rank = np.sum(best_pos <= pos_s) / len(best_pos)  # ранг от 0 до 1
                accepted = (pos_rank >= 0.5) and (best_cls[i] is not None) and (pos_s > neg_s)
                print(f"   Маска {i}: класс={best_cls[i]}, pos={pos_s:.3f}, neg={neg_s:.3f}, diff={conf:+.3f}, rank={pos_rank:.2f}, принято={accepted}")
            else:
                # Стандартная логика
                accepted = (pos_s >= min_pos) and (conf >= threshold) and (best_cls[i] is not None)
                print(f"   Маска {i}: класс={best_cls[i]}, pos={pos_s:.3f}, neg={neg_s:.3f}, diff={conf:+.3f}, принято={accepted}")
            
            if accepted:
                accepted_count += 1
            
            decisions.append({
                'accepted': bool(accepted),
                'class': best_cls[i],
                'pos_score': pos_s,
                'neg_score': neg_s,
                'confidence': float(np.clip(pos_s, 0.0, 1.0))  # совместимо с текущим UI
            })
        
        print(f"🎯 Принято {accepted_count} из {N} масок")
        return decisions
