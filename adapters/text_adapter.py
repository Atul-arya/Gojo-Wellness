"""
Text Adapter - Fractal Analysis for Language/LLM Output

This adapter applies the Universal Filter to text data for:

DETECTION CAPABILITIES:
1. Coherence Analysis: High H = logical structure, low H = word salad
2. Hallucination Detection: Gibberish has different fractal properties
3. Bot vs Human: Writing styles have characteristic fractal signatures
4. Context Importance: High-H paragraphs contain the "skeleton" of logic

THEORY:
- Human writing has long-range correlations (we plan ahead)
- Random text (scrambled) has H ~ 0.5
- Coherent narrative has H > 0.6
- Repetitive/robotic text has low D (simple patterns)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal_core import (
    FractalAnalyzer,
    FractalSignature,
    hurst_exponent,
    higuchi_fractal_dimension,
    SignalType
)
from signal_processor import text_to_signal_simple, text_to_signal_semantic


class TextQuality(Enum):
    """Classification of text quality based on fractal properties."""
    COHERENT = "coherent"           # High H, natural D - well-structured
    RANDOM = "random"               # H ~ 0.5, high D - possibly gibberish
    REPETITIVE = "repetitive"       # Any H, low D - bot-like patterns
    CHAOTIC = "chaotic"             # Low H, very high D - incoherent
    NATURAL = "natural"             # Normal human writing


@dataclass
class TextAnalysis:
    """Complete fractal analysis of text."""
    signature: FractalSignature
    quality: TextQuality
    coherence_score: float      # 0-1, based on H
    complexity_score: float     # 0-1, based on D
    word_count: int
    sentence_count: int
    alerts: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "quality": self.quality.value,
            "coherence_score": round(self.coherence_score, 3),
            "complexity_score": round(self.complexity_score, 3),
            "hurst": round(self.signature.hurst, 4),
            "higuchi": round(self.signature.higuchi, 4),
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "alerts": self.alerts
        }


@dataclass
class ParagraphScore:
    """Importance score for a paragraph."""
    text: str
    index: int
    h_score: float              # Hurst exponent
    importance: str             # "high", "medium", "low"
    keep_for_compression: bool  # Whether to keep in compressed version


class TextAdapter:
    """
    Fractal analysis adapter for text and LLM output.
    
    Analyzes text by converting it to numerical signals and
    measuring fractal properties to assess coherence and quality.
    
    Example:
        >>> adapter = TextAdapter()
        >>> analysis = adapter.analyze("The quick brown fox...")
        >>> print(analysis.quality)  # COHERENT, RANDOM, etc.
    """
    
    # Thresholds for classification
    H_COHERENT = 0.55           # Above this = structured
    H_RANDOM = 0.48             # Below this = potentially random
    
    D_REPETITIVE = 1.25         # Below this = too simple
    D_CHAOTIC = 1.90            # Above this = too random
    
    def __init__(
        self,
        min_words: int = 50,
        signal_method: str = "word_length"
    ):
        """
        Initialize the text adapter.
        
        Args:
            min_words: Minimum words for analysis
            signal_method: How to convert text to signal
                - "word_length": Length of each word
                - "sentence_length": Words per sentence
                - "char_code": Character code sums
        """
        self.min_words = min_words
        self.signal_method = signal_method
        self.analyzer = FractalAnalyzer()
    
    def _preprocess(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def classify_quality(
        self,
        h: float,
        d: float
    ) -> Tuple[TextQuality, List[str]]:
        """
        Classify text quality based on fractal metrics.
        """
        alerts = []
        
        # Check for repetitive (bot-like)
        if d < self.D_REPETITIVE:
            alerts.append(f"[!] Repetitive pattern (D={d:.3f})")
            return TextQuality.REPETITIVE, alerts
        
        # Check for chaotic (incoherent)
        if d >= self.D_CHAOTIC:
            if h < self.H_RANDOM:
                alerts.append(f"[!] Chaotic/incoherent text (H={h:.3f}, D={d:.3f})")
                return TextQuality.CHAOTIC, alerts
        
        # Check for random
        if h < self.H_RANDOM and d > 1.5:
            alerts.append(f"Possibly random text (H={h:.3f})")
            return TextQuality.RANDOM, alerts
        
        # Check for coherent
        if h >= self.H_COHERENT:
            return TextQuality.COHERENT, alerts
        
        return TextQuality.NATURAL, alerts
    
    def analyze(self, text: str) -> TextAnalysis:
        """
        Perform complete fractal analysis on text.
        
        Args:
            text: Input text string
            
        Returns:
            TextAnalysis with all metrics
        """
        text = self._preprocess(text)
        words = text.split()
        
        if len(words) < self.min_words:
            raise ValueError(f"Text too short: {len(words)} words < {self.min_words}")
        
        # Convert to signal
        signal = text_to_signal_simple(text, metric=self.signal_method)
        
        # Analyze
        h, h_conf = hurst_exponent(signal)
        d, d_conf = higuchi_fractal_dimension(signal)
        
        signature = self.analyzer.analyze(signal)
        
        # Classify
        quality, alerts = self.classify_quality(h, d)
        
        # Calculate scores (normalized to 0-1)
        # Coherence: higher H = more coherent
        coherence_score = min(1.0, max(0.0, (h - 0.3) / 0.5))
        
        # Complexity: D in natural range (1.3-1.7) scores highest
        if 1.3 <= d <= 1.7:
            complexity_score = 1.0
        elif d < 1.3:
            complexity_score = max(0.0, (d - 1.0) / 0.3)
        else:
            complexity_score = max(0.0, 1.0 - (d - 1.7) / 0.3)
        
        return TextAnalysis(
            signature=signature,
            quality=quality,
            coherence_score=coherence_score,
            complexity_score=complexity_score,
            word_count=len(words),
            sentence_count=self._count_sentences(text),
            alerts=alerts
        )
    
    def score_paragraphs(
        self,
        text: str,
        h_threshold: float = 0.55
    ) -> List[ParagraphScore]:
        """
        Score individual paragraphs for importance.
        
        Used for the "Fractal Memory" - keeping high-H paragraphs.
        
        Args:
            text: Full text with paragraphs
            h_threshold: H above this = important
            
        Returns:
            List of ParagraphScore for each paragraph
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        scores = []
        
        for i, para in enumerate(paragraphs):
            words = para.split()
            
            if len(words) < 20:
                # Too short to analyze, keep anyway
                scores.append(ParagraphScore(
                    text=para,
                    index=i,
                    h_score=0.5,
                    importance="low",
                    keep_for_compression=len(words) < 10  # Keep very short (might be headers)
                ))
                continue
            
            # Analyze paragraph
            signal = text_to_signal_simple(para, metric=self.signal_method)
            h, _ = hurst_exponent(signal)
            
            if h >= h_threshold:
                importance = "high"
                keep = True
            elif h >= 0.45:
                importance = "medium"
                keep = True  # Keep medium too
            else:
                importance = "low"
                keep = False
            
            scores.append(ParagraphScore(
                text=para,
                index=i,
                h_score=h,
                importance=importance,
                keep_for_compression=keep
            ))
        
        return scores
    
    def compress(
        self,
        text: str,
        keep_ratio: float = 0.5
    ) -> str:
        """
        Compress text by keeping only high-H (structured) paragraphs.
        
        This implements the "Fractal Memory" concept - keeping the
        skeleton of logic while discarding fluff.
        
        Args:
            text: Full text
            keep_ratio: Target ratio of text to keep (0-1)
            
        Returns:
            Compressed text with high-H paragraphs only
        """
        scores = self.score_paragraphs(text)
        
        # Sort by H score (descending)
        sorted_scores = sorted(scores, key=lambda x: x.h_score, reverse=True)
        
        # Calculate how many paragraphs to keep
        total_words = sum(len(s.text.split()) for s in scores)
        target_words = int(total_words * keep_ratio)
        
        kept_words = 0
        kept_indices = set()
        
        for score in sorted_scores:
            if kept_words >= target_words:
                break
            kept_indices.add(score.index)
            kept_words += len(score.text.split())
        
        # Reconstruct in original order
        compressed_paragraphs = [
            scores[i].text for i in sorted(kept_indices)
        ]
        
        return '\n\n'.join(compressed_paragraphs)
    
    def detect_hallucination(
        self,
        text: str,
        reference_h: float = 0.55,
        reference_d: float = 1.5
    ) -> Dict:
        """
        Detect potential LLM hallucination by comparing to expected values.
        
        Hallucinations often have different fractal properties:
        - Lower H (less structured logic)
        - Different D (unusual complexity patterns)
        
        Returns:
            Dictionary with hallucination score and reasoning
        """
        analysis = self.analyze(text)
        
        h_deviation = abs(analysis.signature.hurst - reference_h)
        d_deviation = abs(analysis.signature.higuchi - reference_d)
        
        # Calculate hallucination risk (0-1)
        risk = min(1.0, (h_deviation * 2) + (d_deviation * 0.5))
        
        if risk > 0.5:
            verdict = "HIGH_RISK"
            reason = f"Fractal properties deviate from expected (H_dev={h_deviation:.2f}, D_dev={d_deviation:.2f})"
        elif risk > 0.3:
            verdict = "MEDIUM_RISK"
            reason = "Some deviation from expected patterns"
        else:
            verdict = "LOW_RISK"
            reason = "Fractal properties within normal range"
        
        return {
            "hallucination_risk": round(risk, 3),
            "verdict": verdict,
            "reason": reason,
            "text_h": round(analysis.signature.hurst, 3),
            "text_d": round(analysis.signature.higuchi, 3),
            "expected_h": reference_h,
            "expected_d": reference_d
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEXT ADAPTER - Fractal Language Analysis Demo")
    print("=" * 60)
    
    adapter = TextAdapter()
    
    # 1. Coherent text (should have high H)
    print("\n[1] Coherent Essay")
    coherent_text = """
    The process of photosynthesis is fundamental to life on Earth. Plants use sunlight 
    to convert carbon dioxide and water into glucose and oxygen. This process occurs 
    primarily in the chloroplasts of plant cells. The light-dependent reactions capture 
    solar energy, while the Calvin cycle uses this energy to produce sugars. Without 
    photosynthesis, the oxygen-rich atmosphere we depend on would not exist. Furthermore, 
    the glucose produced serves as the primary energy source for plant growth and 
    development. This intricate biochemical process has evolved over billions of years
    to become remarkably efficient at converting light energy into chemical energy.
    The implications for agriculture, climate science, and renewable energy research
    are profound and continue to drive scientific investigation.
    """
    
    try:
        analysis = adapter.analyze(coherent_text)
        print(f"  Quality: {analysis.quality.value}")
        print(f"  Coherence: {analysis.coherence_score:.2f}, Complexity: {analysis.complexity_score:.2f}")
        print(f"  H={analysis.signature.hurst:.3f}, D={analysis.signature.higuchi:.3f}")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # 2. Random/scrambled text (should have H ~ 0.5)
    print("\n[2] Scrambled Text")
    # Scramble a sentence
    words = coherent_text.split()
    np.random.seed(42)
    np.random.shuffle(words)
    scrambled_text = " ".join(words)
    
    try:
        analysis = adapter.analyze(scrambled_text)
        print(f"  Quality: {analysis.quality.value}")
        print(f"  Coherence: {analysis.coherence_score:.2f}")
        print(f"  H={analysis.signature.hurst:.3f} (expect ~0.5)")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # 3. Repetitive text (should have low D)
    print("\n[3] Repetitive Text")
    repetitive_text = "The cat sat on the mat. " * 30
    
    try:
        analysis = adapter.analyze(repetitive_text)
        print(f"  Quality: {analysis.quality.value}")
        print(f"  D={analysis.signature.higuchi:.3f} (expect low)")
        print(f"  Alerts: {analysis.alerts}")
    except ValueError as e:
        print(f"  Error: {e}")
    
    # 4. Paragraph scoring and compression
    print("\n[4] Paragraph Importance Scoring")
    multi_paragraph = """
    Introduction to the topic. This is some background information about what we will discuss.
    
    The main argument is as follows. First, we observe that X leads to Y. Furthermore, the 
    evidence strongly suggests that Z is a direct consequence of the X-Y relationship. This
    causal chain has been verified in multiple independent studies across different contexts.
    
    Um, yeah, so anyway, we were talking about stuff and things, you know how it is.
    
    In conclusion, the evidence is clear. The relationship between X, Y, and Z represents
    a fundamental principle that has far-reaching implications for our understanding.
    """
    
    scores = adapter.score_paragraphs(multi_paragraph)
    for s in scores:
        print(f"  Para {s.index}: H={s.h_score:.3f} ({s.importance}) - Keep: {s.keep_for_compression}")
    
    print("\n" + "=" * 60)
    print("Text Adapter operational. Ready for analysis.")
    print("=" * 60)
