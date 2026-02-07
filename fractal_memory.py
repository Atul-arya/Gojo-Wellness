"""
Fractal Memory - The Universal Filter for Information Compression

This module implements the "Fractal Memory" concept - a method to compress
information by preserving structural importance rather than repetition.

CORE CONCEPT:
Standard compression (ZIP): Looks for repeated patterns
Fractal compression: Looks for structured/surprising information

THE FILTER:
- "Fluff" (H ~ 0.5): Random entropy, no structure. DELETE.
- "Core" (H > 0.7): Highly structured, causal information. STORE.

APPLICATIONS:
- Compress court transcripts to core facts
- Reduce context windows while preserving logic
- Extract the "skeleton" of any document
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractal_core import hurst_exponent, higuchi_fractal_dimension
from signal_processor import text_to_signal_simple


@dataclass
class MemoryChunk:
    """A chunk of information with its importance score."""
    content: str
    index: int
    h_score: float           # Hurst exponent - structure measure
    d_score: float           # Higuchi dimension - complexity measure
    importance: float        # Combined importance score (0-1)
    category: str            # "core", "supporting", "fluff"
    
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "h_score": round(self.h_score, 4),
            "d_score": round(self.d_score, 4),
            "importance": round(self.importance, 4),
            "category": self.category,
            "content_preview": self.content[:100] + "..." if len(self.content) > 100 else self.content
        }


class FractalMemory:
    """
    Fractal-based information compression system.
    
    Compresses documents by keeping high-H (structured) content
    and discarding low-H (random/fluffy) content.
    
    Example:
        >>> memory = FractalMemory()
        >>> compressed = memory.compress(long_document, ratio=0.3)
        >>> print(f"Reduced to {len(compressed)/len(long_document):.0%}")
    """
    
    # Thresholds for classification
    H_CORE = 0.65             # High structure - definitely keep
    H_SUPPORTING = 0.50       # Medium structure - keep if space
    H_FLUFF = 0.45            # Low structure - safe to discard
    
    D_NATURAL_LOW = 1.25
    D_NATURAL_HIGH = 1.75
    
    def __init__(
        self,
        chunk_method: str = "paragraph",
        min_chunk_words: int = 20
    ):
        """
        Initialize the memory system.
        
        Args:
            chunk_method: How to chunk text ("paragraph", "sentence", "fixed")
            min_chunk_words: Minimum words per chunk for analysis
        """
        self.chunk_method = chunk_method
        self.min_chunk_words = min_chunk_words
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for analysis."""
        if self.chunk_method == "paragraph":
            chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        elif self.chunk_method == "sentence":
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = [s.strip() for s in sentences if s.strip()]
        elif self.chunk_method == "fixed":
            words = text.split()
            chunk_size = 100
            chunks = [' '.join(words[i:i+chunk_size]) 
                     for i in range(0, len(words), chunk_size)]
        else:
            chunks = [text]
        
        return chunks
    
    def _calculate_importance(self, h: float, d: float) -> float:
        """
        Calculate combined importance score from H and D.
        
        Higher H = more structured = more important
        D in natural range = healthy complexity = bonus
        """
        # Base score from Hurst (0 to 0.8 weight)
        h_score = min(1.0, max(0.0, (h - 0.3) / 0.5)) * 0.8
        
        # Bonus from natural complexity (0 to 0.2 weight)
        if self.D_NATURAL_LOW <= d <= self.D_NATURAL_HIGH:
            d_bonus = 0.2
        elif d < self.D_NATURAL_LOW:
            d_bonus = max(0.0, (d - 1.0) / 0.25) * 0.1
        else:
            d_bonus = max(0.0, (2.0 - d) / 0.25) * 0.1
        
        return min(1.0, h_score + d_bonus)
    
    def _categorize(self, h: float) -> str:
        """Categorize chunk based on Hurst exponent."""
        if h >= self.H_CORE:
            return "core"
        elif h >= self.H_SUPPORTING:
            return "supporting"
        else:
            return "fluff"
    
    def analyze_chunks(self, text: str) -> List[MemoryChunk]:
        """
        Analyze all chunks in a document.
        
        Returns:
            List of MemoryChunk with importance scores
        """
        chunks = self._chunk_text(text)
        results = []
        
        for i, chunk in enumerate(chunks):
            words = chunk.split()
            
            if len(words) < self.min_chunk_words:
                # Too short - assign neutral scores
                results.append(MemoryChunk(
                    content=chunk,
                    index=i,
                    h_score=0.5,
                    d_score=1.5,
                    importance=0.3,  # Low but not zero
                    category="fluff" if len(words) > 5 else "supporting"
                ))
                continue
            
            # Convert to signal and analyze
            signal = text_to_signal_simple(chunk, metric="word_length")
            
            try:
                h, _ = hurst_exponent(signal)
                d, _ = higuchi_fractal_dimension(signal)
            except ValueError:
                h, d = 0.5, 1.5
            
            importance = self._calculate_importance(h, d)
            category = self._categorize(h)
            
            results.append(MemoryChunk(
                content=chunk,
                index=i,
                h_score=h,
                d_score=d,
                importance=importance,
                category=category
            ))
        
        return results
    
    def compress(
        self,
        text: str,
        ratio: float = 0.5,
        preserve_order: bool = True
    ) -> str:
        """
        Compress text by keeping only important (high-H) chunks.
        
        Args:
            text: Full document
            ratio: Target compression ratio (0-1)
            preserve_order: Keep chunks in original order
            
        Returns:
            Compressed text
        """
        chunks = self.analyze_chunks(text)
        
        if not chunks:
            return ""
        
        # Sort by importance
        sorted_chunks = sorted(chunks, key=lambda x: x.importance, reverse=True)
        
        # Calculate words to keep
        total_words = sum(len(c.content.split()) for c in chunks)
        target_words = int(total_words * ratio)
        
        # Select chunks
        kept_chunks = []
        kept_words = 0
        
        for chunk in sorted_chunks:
            chunk_words = len(chunk.content.split())
            if kept_words + chunk_words <= target_words or not kept_chunks:
                kept_chunks.append(chunk)
                kept_words += chunk_words
        
        # Restore order if requested
        if preserve_order:
            kept_chunks = sorted(kept_chunks, key=lambda x: x.index)
        
        # Reconstruct
        return '\n\n'.join(c.content for c in kept_chunks)
    
    def extract_skeleton(self, text: str) -> Dict:
        """
        Extract the "skeleton" of a document - just the core facts.
        
        Returns:
            Dictionary with core chunks and statistics
        """
        chunks = self.analyze_chunks(text)
        
        core = [c for c in chunks if c.category == "core"]
        supporting = [c for c in chunks if c.category == "supporting"]
        fluff = [c for c in chunks if c.category == "fluff"]
        
        original_words = sum(len(c.content.split()) for c in chunks)
        core_words = sum(len(c.content.split()) for c in core)
        
        return {
            "skeleton": '\n\n'.join(c.content for c in sorted(core, key=lambda x: x.index)),
            "core_chunks": len(core),
            "supporting_chunks": len(supporting),
            "fluff_chunks": len(fluff),
            "compression_ratio": core_words / original_words if original_words > 0 else 0,
            "categories": {
                "core": [c.to_dict() for c in core],
                "supporting": [c.to_dict() for c in supporting[:5]],  # Limit for brevity
                "fluff": [c.to_dict() for c in fluff[:3]]
            }
        }
    
    def score_for_context_window(
        self,
        text: str,
        max_tokens: int = 4000,
        chars_per_token: int = 4
    ) -> str:
        """
        Compress text to fit within a context window.
        
        Args:
            text: Full document
            max_tokens: Maximum token limit
            chars_per_token: Estimated characters per token
            
        Returns:
            Compressed text fitting within limit
        """
        max_chars = max_tokens * chars_per_token
        
        if len(text) <= max_chars:
            return text
        
        # Calculate required compression ratio
        ratio = max_chars / len(text)
        ratio *= 0.9  # Safety margin
        
        return self.compress(text, ratio=ratio)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FRACTAL MEMORY - Universal Filter Demo")
    print("=" * 60)
    
    memory = FractalMemory()
    
    # Simulated document with varying importance
    document = """
    The defendant arrived at the scene at precisely 4:37 PM on March 15th, 2024.
    Security camera footage clearly shows him entering through the main entrance.
    The timestamp is corroborated by the building's electronic access logs.

    So like, yeah, the weather was pretty nice that day I guess. People were 
    walking around and stuff. Nothing special really happening outside. Just
    a normal Tuesday afternoon, you know how it is. Birds were chirping.

    Critical evidence was recovered from the third floor office. The forensic
    analysis revealed fingerprints matching the defendant on the cabinet handle.
    DNA evidence collected from a coffee cup confirmed the defendant's presence.
    These findings were independently verified by two separate laboratories.

    Um, let me think... there were some other people around maybe? I'm not really
    sure about the details. It was a while ago and my memory isn't the best.
    Could have been anyone really. Hard to say for certain.

    The financial records show a transfer of $2.4 million to an offshore account
    on the same day. Transaction ID: TXN-2024-03-15-8847392. The account was
    opened just three days prior using forged identification documents.
    """
    
    print("\n[1] Chunk Analysis")
    chunks = memory.analyze_chunks(document)
    for c in chunks:
        preview = c.content[:50].replace('\n', ' ')
        print(f"  Para {c.index}: H={c.h_score:.2f}, Category={c.category:10} | '{preview}...'")
    
    print("\n[2] Extract Skeleton (Core Facts Only)")
    skeleton = memory.extract_skeleton(document)
    print(f"  Core chunks: {skeleton['core_chunks']}")
    print(f"  Fluff chunks: {skeleton['fluff_chunks']}")
    print(f"  Compression: {skeleton['compression_ratio']:.0%}")
    
    print("\n[3] Compressed to 50%")
    compressed = memory.compress(document, ratio=0.5)
    original_len = len(document)
    compressed_len = len(compressed)
    print(f"  Original: {original_len} chars")
    print(f"  Compressed: {compressed_len} chars ({compressed_len/original_len:.0%})")
    print(f"\n--- Compressed Content ---")
    print(compressed[:500] + "..." if len(compressed) > 500 else compressed)
    
    print("\n" + "=" * 60)
    print("Fractal Memory operational. Information skeleton extracted.")
    print("=" * 60)


class TimeFractalMemory:
    """
    Fractal compression for time-series data (e.g., biometric logs).
    
    Keeps "interesting" moments (high H / stress) and summarizes routine data.
    """
    
    def compress_history(self, history: List[Dict], ratio: float = 0.1) -> List[Dict]:
        """
        Compress a list of state snapshots.
        
        Args:
            history: List of StateSnapshot dicts
            ratio: Target retention ratio (approximate)
            
        Returns:
            Compressed history list
        """
        if not history:
            return []
            
        # 1. Identify "events" vs "routine"
        # We process in chunks of 60 mins (approx 120 samples if 30s interval)
        # But here we just stream through
        
        compressed = []
        buffer = []
        
        for entry in history:
            # Always keep high stress or deviation events
            is_significant = (
                entry.get('stress_relative', 0) > 0.4 or 
                entry.get('deviation') == 'elevated' or
                entry.get('h', 0.5) > 0.7  # High structure/novelty
            )
            
            if is_significant:
                # Flush buffer as summary if exists
                if buffer:
                    compressed.append(self._summarize_buffer(buffer))
                    buffer = []
                compressed.append(entry)
            else:
                buffer.append(entry)
                
                # If buffer gets too big (e.g. 1 hour of routine), summarize it
                if len(buffer) >= 20:  # ~10-20 mins of data
                    compressed.append(self._summarize_buffer(buffer))
                    buffer = []
        
        # Flush remaining
        if buffer:
            compressed.append(self._summarize_buffer(buffer))
            
        return compressed

    def _summarize_buffer(self, buffer: List[Dict]) -> Dict:
        """Create a summary entry from a buffer of routine data."""
        if not buffer:
            return {}
            
        # Use the middle timestamp
        mid_idx = len(buffer) // 2
        summary = buffer[mid_idx].copy()
        
        # Requests "Average" values
        h_vals = [d.get('h', 0.5) for d in buffer]
        d_vals = [d.get('d', 1.5) for d in buffer]
        stress_vals = [d.get('stress_relative', 0) for d in buffer]
        
        summary['h'] = round(float(np.mean(h_vals)), 4)
        summary['d'] = round(float(np.mean(d_vals)), 4)
        summary['stress_relative'] = round(float(np.mean(stress_vals)), 3)
        summary['state'] = "routine_summary"
        summary['deviation'] = "normal"
        summary['note'] = f"Summary of {len(buffer)} routine points"
        
        return summary
    
    def search_pattern(self, history: List[Dict], target_h: float, target_d: float, tolerance: float = 0.1) -> List[Dict]:
        """
        Search history for moments with similar fractal structure.
        
        Args:
            history: List of StateSnapshot dicts (can be compressed)
            target_h: Target Hurst exponent
            target_d: Target Higuchi dimension
            tolerance: Euclidean distance threshold
            
        Returns:
            List of matching snapshots
        """
        matches = []
        
        for entry in history:
            h = entry.get('h', 0.5)
            d = entry.get('d', 1.5)
            
            # Simple Euclidean distance in H-D space
            # Normalize axes roughly (H: 0-1, D: 1-2) so they have equal weight
            dist = np.sqrt((h - target_h)**2 + (d - target_d)**2)
            
            if dist <= tolerance:
                entry_copy = entry.copy()
                entry_copy['match_score'] = round(1.0 - dist, 2)
                matches.append(entry_copy)
                
        # Sort by best match
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)


