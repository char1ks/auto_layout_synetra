"""
Этап 5: СКОРИНГ И ПРИНЯТИЕ РЕШЕНИЙ

Основная задача: Для каждой маски-кандидата вычислить оценку (confidence) 
и принять решение, является ли она искомым объектом.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

from ..utils.config import ScoringConfig


class ScoringEngine:
    """Класс для скоринга масок и принятия решений."""
    
    def __init__(self, config: ScoringConfig):
        """
        Инициализация движка скоринга.
        
        Args:
            config: Конфигурация скоринга
        """
        self.config = config
    
    def score_and_decide(self, mask_embeddings: np.ndarray,
                        positive_embeddings: np.ndarray,
                        negative_embeddings: np.ndarray) -> Tuple[List[bool], List[float], List[float], List[float]]:
        """
        Выполняет скоринг и принятие решений для всех масок.
        
        Args:
            mask_embeddings: Эмбеддинги масок [N_masks, dim]
            positive_embeddings: Эмбеддинги positive примеров [N_pos, dim] 
            negative_embeddings: Эмбеддинги negative примеров [N_neg, dim]
            
        Returns:
            Кортеж (accept_flags, confidence_scores, positive_scores, negative_scores)
        """
        print("\n🔄 ЭТАП 5: СКОРИНГ И ПРИНЯТИЕ РЕШЕНИЙ")
        print("=" * 60)
        
        if mask_embeddings.size == 0:
            print("   ⚠️ Нет эмбеддингов масок для скоринга")
            return [], [], [], []
        
        if positive_embeddings.size == 0:
            print("   ⚠️ Нет positive примеров для сравнения")
            return [], [], [], []
        
        print(f"   📊 Скоринг {len(mask_embeddings)} масок против {len(positive_embeddings)} positive примеров")
        if len(negative_embeddings) > 0:
            print(f"   📊 Учитываем {len(negative_embeddings)} negative примеров")
        
        # 5.1. Вычисление матриц сходства
        positive_similarities, negative_similarities = self._compute_similarity_matrices(
            mask_embeddings, positive_embeddings, negative_embeddings
        )
        
        # 5.2. Агрегация positive скоров
        positive_scores = self._aggregate_positive_scores(positive_similarities)
        
        # 5.3. Агрегация negative скоров (если есть)
        negative_scores = self._aggregate_negative_scores(negative_similarities)
        
        # 5.4. Проверка консенсуса
        consensus_flags = self._check_consensus(positive_similarities)
        
        # 5.5. Применение правил принятия решений
        accept_flags, confidence_scores = self._apply_acceptance_rules(
            positive_scores, negative_scores, consensus_flags
        )
        
        accepted_count = sum(accept_flags)
        print(f"   ✅ Результат скоринга: {accepted_count}/{len(mask_embeddings)} масок принято")
        
        return accept_flags, confidence_scores, positive_scores, negative_scores
    
    def _compute_similarity_matrices(self, mask_embeddings: np.ndarray,
                                    positive_embeddings: np.ndarray,
                                    negative_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        5.1. Вычисление матриц косинусного сходства.
        
        Args:
            mask_embeddings: Эмбеддинги масок
            positive_embeddings: Эмбеддинги positive примеров
            negative_embeddings: Эмбеддинги negative примеров
            
        Returns:
            Кортеж (positive_similarities, negative_similarities)
        """
        print("   🧮 Вычисление матриц сходства...")
        
        # Косинусное сходство = скалярное произведение нормализованных векторов
        # Поскольку векторы уже L2-нормализованы, просто умножаем матрицы
        positive_similarities = self._cosine_matrix(mask_embeddings, positive_embeddings)
        
        negative_similarities = np.array([])
        if negative_embeddings.size > 0:
            negative_similarities = self._cosine_matrix(mask_embeddings, negative_embeddings)
        
        print(f"     ✅ Positive сходство: {positive_similarities.shape}")
        if negative_similarities.size > 0:
            print(f"     ✅ Negative сходство: {negative_similarities.shape}")
        
        return positive_similarities, negative_similarities
    
    def _cosine_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу косинусного сходства между векторами A и B.
        
        Args:
            A: Массив векторов [N, dim]
            B: Массив векторов [M, dim]
            
        Returns:
            Матрица сходства [N, M]
        """
        # A @ B.T для L2-нормализованных векторов даёт косинусное сходство
        return A @ B.T
    
    def _aggregate_positive_scores(self, positive_similarities: np.ndarray) -> np.ndarray:
        """
        5.2. Агрегация positive скоров.
        
        Args:
            positive_similarities: Матрица сходства с positive примерами [N_masks, N_pos]
            
        Returns:
            Массив positive скоров [N_masks]
        """
        if positive_similarities.size == 0:
            return np.array([])
        
        n_masks = positive_similarities.shape[0]
        positive_scores = np.zeros(n_masks)
        
        for i in range(n_masks):
            similarities = positive_similarities[i]  # сходства для i-ой маски
            
            # Максимальное сходство
            max_sim = np.max(similarities)
            
            # Top-k среднее (берём лучшие k сходств)
            k = min(len(similarities), 3)  # например, топ-3
            topk_indices = np.argpartition(similarities, -k)[-k:]
            topk_mean = np.mean(similarities[topk_indices])
            
            # Взвешенная комбинация: 70% максимум + 30% топ-k среднее
            combined_score = 0.7 * max_sim + 0.3 * topk_mean
            
            # Переводим из диапазона [-1, 1] в [0, 1]
            normalized_score = (combined_score + 1.0) * 0.5
            
            positive_scores[i] = normalized_score
        
        print(f"     📈 Positive скоры: min={positive_scores.min():.3f}, max={positive_scores.max():.3f}, mean={positive_scores.mean():.3f}")
        return positive_scores
    
    def _aggregate_negative_scores(self, negative_similarities: np.ndarray) -> np.ndarray:
        """
        Агрегация negative скоров.
        
        Args:
            negative_similarities: Матрица сходства с negative примерами [N_masks, N_neg]
            
        Returns:
            Массив negative скоров [N_masks]
        """
        if negative_similarities.size == 0:
            return np.array([])
        
        # Для negative примеров берём максимальное сходство (худший случай)
        max_negative_similarities = np.max(negative_similarities, axis=1)
        
        # Переводим в диапазон [0, 1]
        negative_scores = (max_negative_similarities + 1.0) * 0.5
        
        print(f"     📉 Negative скоры: min={negative_scores.min():.3f}, max={negative_scores.max():.3f}, mean={negative_scores.mean():.3f}")
        return negative_scores
    
    def _check_consensus(self, positive_similarities: np.ndarray) -> np.ndarray:
        """
        5.3. Проверка консенсуса positive примеров.
        
        Args:
            positive_similarities: Матрица сходства с positive примерами
            
        Returns:
            Массив флагов консенсуса [N_masks]
        """
        if positive_similarities.size == 0:
            return np.array([])
        
        n_masks = positive_similarities.shape[0]
        consensus_flags = np.zeros(n_masks, dtype=bool)
        
        for i in range(n_masks):
            similarities = positive_similarities[i]
            
            # Считаем количество positive примеров с высоким сходством
            high_similarity_count = np.sum(similarities > self.config.consensus_thr)
            
            # Консенсус достигнут если минимальное количество примеров согласны
            consensus_flags[i] = high_similarity_count >= self.config.consensus_k
        
        consensus_count = np.sum(consensus_flags)
        print(f"     🤝 Консенсус (>{self.config.consensus_thr:.2f}, k>={self.config.consensus_k}): {consensus_count}/{n_masks} масок")
        
        return consensus_flags
    
    def _apply_acceptance_rules(self, positive_scores: np.ndarray,
                               negative_scores: np.ndarray,
                               consensus_flags: np.ndarray) -> Tuple[List[bool], List[float]]:
        """
        5.4. Применение финальных правил принятия решений.
        
        Args:
            positive_scores: Positive скоры
            negative_scores: Negative скоры  
            consensus_flags: Флаги консенсуса
            
        Returns:
            Кортеж (accept_flags, confidence_scores)
        """
        if len(positive_scores) == 0:
            return [], []
        
        n_masks = len(positive_scores)
        accept_flags = []
        confidence_scores = []
        
        # Если нет negative скоров, создаём массив нулей
        if len(negative_scores) == 0:
            negative_scores = np.zeros(n_masks)
        
        # Если нет флагов консенсуса, считаем что консенсус есть всегда
        if len(consensus_flags) == 0:
            consensus_flags = np.ones(n_masks, dtype=bool)
        
        eps = 1e-8  # для избежания деления на ноль
        
        for i in range(n_masks):
            pos_score = positive_scores[i]
            neg_score = negative_scores[i] if i < len(negative_scores) else 0.0
            has_consensus = consensus_flags[i] if i < len(consensus_flags) else True
            
            # Применяем все правила приёма
            accept = True
            
            # Правило 1: Минимальная уверенность
            if pos_score < self.config.min_confidence:
                accept = False
            
            # Правило 2: Минимальная разница между positive и negative
            margin = pos_score - neg_score
            if margin < self.config.margin:
                accept = False
            
            # Правило 3: Минимальное отношение positive к negative
            ratio = pos_score / (neg_score + eps)
            if ratio < self.config.ratio:
                accept = False
            
            # Правило 4: Максимальная negative оценка
            if neg_score > self.config.neg_cap:
                accept = False
            
            # Правило 5: Консенсус
            if not has_consensus:
                accept = False
            
            accept_flags.append(accept)
            confidence_scores.append(pos_score)  # Используем positive скор как confidence
        
        # Логирование статистики
        accepted_count = sum(accept_flags)
        if accepted_count > 0:
            accepted_confidences = [confidence_scores[i] for i, acc in enumerate(accept_flags) if acc]
            print(f"     ✅ Принято {accepted_count} масок с confidence: {np.mean(accepted_confidences):.3f}±{np.std(accepted_confidences):.3f}")
        else:
            print(f"     ❌ Ни одна маска не прошла все правила приёма")
        
        # Подробная статистика правил
        self._log_rule_statistics(positive_scores, negative_scores, consensus_flags, accept_flags)
        
        return accept_flags, confidence_scores
    
    def _log_rule_statistics(self, positive_scores: np.ndarray, negative_scores: np.ndarray,
                           consensus_flags: np.ndarray, accept_flags: List[bool]):
        """
        Логирует подробную статистику применения правил.
        
        Args:
            positive_scores: Positive скоры
            negative_scores: Negative скоры
            consensus_flags: Флаги консенсуса
            accept_flags: Финальные флаги принятия
        """
        n_masks = len(positive_scores)
        if n_masks == 0:
            return
        
        # Если нет negative скоров, создаём массив нулей
        if len(negative_scores) == 0:
            negative_scores = np.zeros(n_masks)
        
        eps = 1e-8
        
        # Подсчёт нарушений каждого правила
        confidence_fails = sum(1 for score in positive_scores if score < self.config.min_confidence)
        
        margin_fails = sum(1 for i in range(n_masks) 
                          if (positive_scores[i] - negative_scores[i]) < self.config.margin)
        
        ratio_fails = sum(1 for i in range(n_masks) 
                         if (positive_scores[i] / (negative_scores[i] + eps)) < self.config.ratio)
        
        neg_cap_fails = sum(1 for score in negative_scores if score > self.config.neg_cap)
        
        consensus_fails = sum(1 for flag in consensus_flags if not flag)
        
        print(f"     📋 Статистика правил приёма:")
        print(f"       • Confidence >={self.config.min_confidence:.2f}: {n_masks - confidence_fails}/{n_masks} пройдено")
        print(f"       • Margin >={self.config.margin:.2f}: {n_masks - margin_fails}/{n_masks} пройдено")
        print(f"       • Ratio >={self.config.ratio:.1f}: {n_masks - ratio_fails}/{n_masks} пройдено")
        print(f"       • Neg cap <={self.config.neg_cap:.2f}: {n_masks - neg_cap_fails}/{n_masks} пройдено")
        print(f"       • Consensus: {n_masks - consensus_fails}/{n_masks} пройдено")
    
    def get_scoring_summary(self, accept_flags: List[bool], confidence_scores: List[float],
                           positive_scores: List[float], negative_scores: List[float]) -> Dict[str, Any]:
        """
        Возвращает сводку результатов скоринга.
        
        Args:
            accept_flags: Флаги принятия
            confidence_scores: Скоры уверенности  
            positive_scores: Positive скоры
            negative_scores: Negative скоры
            
        Returns:
            Словарь со сводкой
        """
        if not accept_flags:
            return {"total_masks": 0, "accepted_masks": 0}
        
        accepted_indices = [i for i, flag in enumerate(accept_flags) if flag]
        
        summary = {
            "total_masks": len(accept_flags),
            "accepted_masks": len(accepted_indices),
            "acceptance_rate": len(accepted_indices) / len(accept_flags),
            "mean_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "mean_positive_score": np.mean(positive_scores) if positive_scores else 0,
            "mean_negative_score": np.mean(negative_scores) if negative_scores else 0,
        }
        
        if accepted_indices:
            accepted_confidences = [confidence_scores[i] for i in accepted_indices]
            summary.update({
                "accepted_mean_confidence": np.mean(accepted_confidences),
                "accepted_min_confidence": np.min(accepted_confidences),
                "accepted_max_confidence": np.max(accepted_confidences),
            })
        
        return summary
