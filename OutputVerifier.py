# Output verification: computes loss/similarity between predicted and ground truth grids

import numpy as np
from typing import List, Tuple


class OutputVerifier:
    # Computes loss and similarity metrics between grids
    
    def compute_loss(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        # Loss is normalized Hamming distance [0, 1] where 0 = perfect match
        # print(f"[OutputVerifier] Computing loss between {predicted.shape} and {ground_truth.shape} grids")
        # Handle shape mismatch
        if predicted.shape != ground_truth.shape:
            min_h = min(predicted.shape[0], ground_truth.shape[0])
            min_w = min(predicted.shape[1], ground_truth.shape[1])
            predicted = predicted[:min_h, :min_w]
            ground_truth = ground_truth[:min_h, :min_w]
        
        # Pixel-wise absolute difference
        diff = np.abs(predicted.astype(np.int32) - ground_truth.astype(np.int32))
        total_error = np.sum(diff)
        
        # Normalize: max error is all pixels with value 9 different
        num_pixels = predicted.size
        max_error = num_pixels * 9
        
        loss = float(total_error) / float(max_error) if max_error > 0 else 0.0
        # print(f"[OutputVerifier] Loss: {loss:.4f} (error: {total_error}, max: {max_error})")
        return np.clip(loss, 0.0, 1.0)
    
    def pixel_accuracy(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        # Compute percentage of pixels that match exactly [0, 1]
        # print(f"[OutputVerifier] Computing pixel accuracy")
        if predicted.shape != ground_truth.shape:
            min_h = min(predicted.shape[0], ground_truth.shape[0])
            min_w = min(predicted.shape[1], ground_truth.shape[1])
            predicted = predicted[:min_h, :min_w]
            ground_truth = ground_truth[:min_h, :min_w]
        
        matches = np.sum(predicted == ground_truth)
        total = predicted.size
        
        return float(matches) / float(total) if total > 0 else 0.0
        # print(f"[OutputVerifier] Pixel accuracy: {matches}/{total} = {float(matches)/float(total):.4f}")
    
    def structural_similarity(self, predicted: np.ndarray, ground_truth: np.ndarray, 
                             window_size: int = 5) -> float:
        # Compute structural similarity using local pattern comparison
        # print(f"[OutputVerifier] Computing SSIM with window size {window_size}")
        if predicted.shape != ground_truth.shape:
            min_h = min(predicted.shape[0], ground_truth.shape[0])
            min_w = min(predicted.shape[1], ground_truth.shape[1])
            predicted = predicted[:min_h, :min_w]
            ground_truth = ground_truth[:min_h, :min_w]
        
        h, w = predicted.shape
        similarities = []
        
        for r in range(0, h - window_size + 1, window_size):
            for c in range(0, w - window_size + 1, window_size):
                pred_patch = predicted[r:r+window_size, c:c+window_size]
                truth_patch = ground_truth[r:r+window_size, c:c+window_size]
                
                # Simple patch similarity: percentage of matching pixels
                match = np.sum(pred_patch == truth_patch)
                total = window_size * window_size
                patch_sim = float(match) / float(total)
                similarities.append(patch_sim)
        
        if similarities:
            result = float(np.mean(similarities))
            # print(f"[OutputVerifier] SSIM: {result:.4f}")
            return result
        else:
            # Grids too small for window
            return self.pixel_accuracy(predicted, ground_truth)
    
    def object_wise_accuracy(self, predicted: np.ndarray, ground_truth: np.ndarray, 
                            num_colors: int = 10) -> Tuple[float, int, int]:
        # Compute accuracy by color (object-wise)
        # print(f"[OutputVerifier] Computing object-wise accuracy for {num_colors} colors")
        if predicted.shape != ground_truth.shape:
            min_h = min(predicted.shape[0], ground_truth.shape[0])
            min_w = min(predicted.shape[1], ground_truth.shape[1])
            predicted = predicted[:min_h, :min_w]
            ground_truth = ground_truth[:min_h, :min_w]
        
        total_correct = 0
        total_pixels = 0
        
        for color in range(num_colors):
            pred_mask = (predicted == color)
            truth_mask = (ground_truth == color)
            
            # Both predict and ground truth have this color at same location
            correct = np.sum(pred_mask & truth_mask)
            total = np.sum(truth_mask)  # Count ground truth pixels
            
            total_correct += correct
            total_pixels += total
        
        accuracy = float(total_correct) / float(total_pixels) if total_pixels > 0 else 0.0
        # print(f"[OutputVerifier] Object-wise accuracy: {accuracy:.4f}")
        return accuracy, total_correct, total_pixels


class CandidateRanker:
    # Rank candidate outputs by similarity to ground truth
    
    def __init__(self, verifier: OutputVerifier):
        self.verifier = verifier
    
    def rank(self, candidates: List[Tuple[np.ndarray, float]], 
            ground_truth: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        # Rank candidates by similarity to ground truth
        # print(f"[CandidateRanker] Ranking {len(candidates)} candidates")
        ranked = []
        
        for output, initial_score in candidates:
            # Compute similarity metrics
            loss = self.verifier.compute_loss(output, ground_truth)
            pixel_acc = self.verifier.pixel_accuracy(output, ground_truth)
            structural = self.verifier.structural_similarity(output, ground_truth)
            
            # Combined score: prefer high similarity and low loss
            combined = (pixel_acc + structural) / 2.0
            ranked.append((output, initial_score, combined))
        
        # Sort by combined score (descending)
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        return ranked
