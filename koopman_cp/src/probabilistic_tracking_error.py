#!/usr/bin/env python3
"""
Probabilistic Tracking Error Calculator for Conformal Koopman Control

This module implements the probabilistic tracking error bounds described in the
"Distribution-Free Control and Planning with Koopman" framework.

Based on the mathematical framework from:
Hiroyasu Tsukamoto, UIUC Aerospace, "Distribution-Free Koopman", April 2025

Key concepts implemented:
1. High-Probability Exponential Boundedness using conformal prediction
2. Forward embedding nonconformity scores
3. Inverse embedding nonconformity scores  
4. Constraint tightening for probabilistic guarantees
"""

import json
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Optional imports
try:
    from scipy.linalg import solve_discrete_are
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ConformalKoopmanParams:
    """Parameters for conformal Koopman tracking error calculation."""
    alpha: float = 0.1  # Confidence level for forward embedding (1-alpha confidence)
    beta: float = 0.1   # Confidence level for inverse embedding (1-beta confidence)
    gamma: float = 0.9  # Lyapunov contraction rate
    rho: float = 0.01   # Robustification constant
    cv: float = 1.0     # Weight for Lyapunov modeling error
    K: int = 100        # Prediction horizon


@dataclass  
class TrackingErrorBounds:
    """Results of probabilistic tracking error calculation."""
    forward_quantile: float
    inverse_quantile: float
    delta_r: float  # High-probability tracking error bound
    prob_bound_lower: float  # Lower bound on probability (1-alpha-beta)
    lyapunov_bound: float  # Lyapunov-based bound
    exponential_decay_rate: float  # gamma^K


class ProbabilisticTrackingError:
    """
    Calculate probabilistic tracking error bounds using conformal prediction
    for Koopman-based control systems.
    """
    
    def __init__(self, params: ConformalKoopmanParams):
        self.params = params
    
    def load_trajectory_data(self, file_path: str) -> Dict:
        """Load trajectory data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def extract_poses_and_targets(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pose and target data from JSON.
        
        Returns:
            poses: (N, 3) array of actual positions
            targets: (N, 3) array of target positions  
            times: (N,) array of timestamps
        """
        poses = np.array(data.get('pose', []))
        targets = np.array(data.get('target', []))
        times = np.array(data.get('time', []))
        
        # Ensure poses is properly shaped
        if len(poses.shape) == 2 and poses.shape[1] >= 3:
            poses = poses[:, :3]  # Take only x, y, z
        elif len(poses.shape) == 1:
            poses = poses.reshape(-1, 3) if len(poses) % 3 == 0 else poses.reshape(-1, 1)
            
        # Handle targets
        if len(targets) == 0:
            # If no targets provided, generate ideal circular trajectory as reference
            targets = self.generate_circular_reference(poses, times)
        elif len(targets.shape) == 2 and targets.shape[1] >= 3:
            targets = targets[:, :3]  # Take only x, y, z
        elif len(targets.shape) == 1:
            targets = targets.reshape(-1, 3) if len(targets) % 3 == 0 else targets.reshape(-1, 1)
            
        return poses, targets, times
    
    def generate_circular_reference(self, poses: np.ndarray, times: np.ndarray, 
                                  radius: float = 0.5) -> np.ndarray:
        """
        Generate ideal circular trajectory as reference targets.
        
        Args:
            poses: Actual pose data to determine center and trajectory pattern
            times: Time array
            radius: Radius for circular trajectory
            
        Returns:
            targets: (N, 3) array of ideal circular positions
        """
        if len(poses) == 0:
            return np.array([])
            
        # Estimate center from pose data
        center = np.mean(poses, axis=0)
        
        # Generate circular trajectory
        if len(times) > 0:
            time_steps = times - times[0]  # Start from t=0
        else:
            time_steps = np.arange(len(poses)) * 0.1  # Assume 0.1s intervals
            
        # Estimate angular frequency from data
        if len(poses) > 10:
            # Calculate rough angular velocity from first and last 10% of trajectory
            start_poses = poses[:len(poses)//10]
            end_poses = poses[-len(poses)//10:]
            
            start_angle = np.mean([math.atan2(p[1] - center[1], p[0] - center[0]) for p in start_poses])
            end_angle = np.mean([math.atan2(p[1] - center[1], p[0] - center[0]) for p in end_poses])
            
            angle_diff = end_angle - start_angle
            # Handle wraparound
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
                
            total_time = time_steps[-1] if len(time_steps) > 0 else len(poses) * 0.1
            omega = angle_diff / total_time if total_time > 0 else 0.1  # rad/s
        else:
            omega = 0.1  # Default angular velocity
        
        # Generate ideal circular trajectory
        targets = np.zeros_like(poses)
        for i, t in enumerate(time_steps):
            angle = omega * t
            targets[i, 0] = center[0] + radius * math.cos(angle)
            targets[i, 1] = center[1] + radius * math.sin(angle)
            targets[i, 2] = center[2]  # Keep z constant
            
        return targets
    
    def detect_liftoff_end(self, targets: np.ndarray) -> int:
        """
        Detect when liftoff phase ends based on target trajectory.
        
        Args:
            targets: (N, 3) array of target positions
            
        Returns:
            Index where tracking phase begins
        """
        liftoff_end_index = 0
        for i, target in enumerate(targets):
            if len(target) >= 3 and not np.allclose(target, [0.0, 1.0, 0.0]):
                liftoff_end_index = i
                break
        return liftoff_end_index
    
    def calculate_tracking_errors(self, poses: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculate tracking errors between poses and targets.
        
        Args:
            poses: (N, 3) actual positions
            targets: (N, 3) target positions
            
        Returns:
            tracking_errors: (N,) array of Euclidean tracking errors
        """
        if poses.shape != targets.shape:
            min_len = min(len(poses), len(targets))
            poses = poses[:min_len]
            targets = targets[:min_len]
        
        tracking_errors = np.linalg.norm(poses - targets, axis=1)
        return tracking_errors
    
    def calculate_forward_nonconformity_scores(self, tracking_errors: np.ndarray, 
                                             delta_v: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate forward embedding nonconformity scores.
        
        s_fwd^(i) = Δv,k + Δd,k
        
        Args:
            tracking_errors: Array of tracking errors (Δd,k)
            delta_v: Array of Lyapunov modeling errors (optional)
            
        Returns:
            Forward nonconformity scores
        """
        if delta_v is None:
            delta_v = np.zeros_like(tracking_errors)
        
        return delta_v + tracking_errors
    
    def calculate_inverse_nonconformity_scores(self, poses: np.ndarray, 
                                             predicted_poses: np.ndarray) -> np.ndarray:
        """
        Calculate inverse embedding nonconformity scores.
        
        s_inv^(i) = ||x^(i) - ĝ_inv(z^(i))||
        
        Args:
            poses: (N, 3) actual positions
            predicted_poses: (N, 3) predicted positions from inverse embedding
            
        Returns:
            Inverse nonconformity scores
        """
        return np.linalg.norm(poses - predicted_poses, axis=1)
    
    def compute_quantiles(self, scores: np.ndarray, alpha: float) -> float:
        """
        Compute (1-alpha) quantile of nonconformity scores.
        
        Args:
            scores: Array of nonconformity scores
            alpha: Confidence level
            
        Returns:
            (1-alpha) quantile value
        """
        if len(scores) == 0:
            return 0.0
        return np.quantile(scores, 1 - alpha)
    
    def calculate_lyapunov_based_bound(self, v0: float, forward_quantile: float) -> float:
        """
        Calculate Lyapunov-based tracking error bound.
        
        Based on: v_K ≤ γ^K * v0 + C * (1-γ^K)/(1-γ)
        where C = -ρ + q_fwd(1-α/K)
        
        Args:
            v0: Initial tracking error
            forward_quantile: Forward embedding quantile q_fwd(1-α/K)
            
        Returns:
            Lyapunov bound on tracking error
        """
        gamma = self.params.gamma
        rho = self.params.rho
        K = self.params.K
        
        C = -rho + forward_quantile
        
        if gamma == 1.0:
            lyapunov_bound = v0 + C * K
        else:
            lyapunov_bound = (gamma ** K) * v0 + C * (1 - gamma ** K) / (1 - gamma)
        
        return max(0.0, lyapunov_bound)
    
    def calculate_delta_r(self, forward_quantile: float, m: float = 1.0) -> float:
        """
        Calculate Δr bound for high-probability tracking error.
        
        Δr = C * (1-γ)^(-1) / √m
        where C = -ρ + q_fwd(1-α/K)
        
        Args:
            forward_quantile: Forward embedding quantile
            m: Lower bound on eigenvalues of M matrix
            
        Returns:
            Delta r bound
        """
        gamma = self.params.gamma
        rho = self.params.rho
        
        C = -rho + forward_quantile
        delta_r = C / ((1 - gamma) * math.sqrt(m))
        
        return max(0.0, delta_r)
    
    def analyze_single_trajectory(self, file_path: str, 
                                predicted_poses: Optional[np.ndarray] = None) -> TrackingErrorBounds:
        """
        Analyze a single trajectory file for probabilistic tracking error.
        
        Args:
            file_path: Path to JSON trajectory file
            predicted_poses: Optional predicted poses for inverse embedding analysis
            
        Returns:
            TrackingErrorBounds object with results
        """
        # Load and process data
        data = self.load_trajectory_data(file_path)
        poses, targets, times = self.extract_poses_and_targets(data)
        
        # Remove liftoff phase
        liftoff_end = self.detect_liftoff_end(targets)
        poses = poses[liftoff_end:]
        targets = targets[liftoff_end:]
        
        # Calculate tracking errors
        tracking_errors = self.calculate_tracking_errors(poses, targets)
        
        # Forward embedding analysis
        forward_scores = self.calculate_forward_nonconformity_scores(tracking_errors)
        alpha_K = self.params.alpha / self.params.K
        forward_quantile = self.compute_quantiles(forward_scores, alpha_K)
        
        # Inverse embedding analysis (use tracking errors as approximation if no predicted poses)
        if predicted_poses is not None:
            inverse_scores = self.calculate_inverse_nonconformity_scores(poses, predicted_poses)
        else:
            # Approximate inverse scores using tracking errors
            inverse_scores = tracking_errors
        
        inverse_quantile = self.compute_quantiles(inverse_scores, self.params.beta)
        
        # Calculate bounds
        v0 = tracking_errors[0] if len(tracking_errors) > 0 else 0.0
        lyapunov_bound = self.calculate_lyapunov_based_bound(v0, forward_quantile)
        delta_r = self.calculate_delta_r(forward_quantile)
        
        # Probability bound
        prob_bound_lower = 1 - self.params.alpha - self.params.beta
        
        return TrackingErrorBounds(
            forward_quantile=forward_quantile,
            inverse_quantile=inverse_quantile,
            delta_r=delta_r,
            prob_bound_lower=prob_bound_lower,
            lyapunov_bound=lyapunov_bound,
            exponential_decay_rate=self.params.gamma ** self.params.K
        )
    
    def compare_trajectories(self, file_paths: List[str]) -> Dict[str, TrackingErrorBounds]:
        """
        Compare probabilistic tracking error bounds across multiple trajectories.
        
        Args:
            file_paths: List of paths to trajectory JSON files
            
        Returns:
            Dictionary mapping file names to TrackingErrorBounds
        """
        results = {}
        for file_path in file_paths:
            file_name = Path(file_path).name
            try:
                bounds = self.analyze_single_trajectory(file_path)
                results[file_name] = bounds
            except Exception as e:
                print(f"Error analyzing {file_name}: {e}")
                
        return results
    
    def plot_tracking_error_analysis(self, file_path: str, save_path: Optional[str] = None):
        """
        Plot tracking error analysis results.
        
        Args:
            file_path: Path to trajectory JSON file
            save_path: Optional path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping plot generation.")
            return
        # Load and process data
        data = self.load_trajectory_data(file_path)
        poses, targets, times = self.extract_poses_and_targets(data)
        
        # Remove liftoff phase
        liftoff_end = self.detect_liftoff_end(targets)
        poses = poses[liftoff_end:]
        targets = targets[liftoff_end:]
        times = times[liftoff_end:] if len(times) > 0 else np.arange(len(poses))
        
        # Calculate tracking errors and bounds
        tracking_errors = self.calculate_tracking_errors(poses, targets)
        bounds = self.analyze_single_trajectory(file_path)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot tracking errors over time
        ax1.plot(times, tracking_errors, 'b-', label='Actual Tracking Error', linewidth=2)
        ax1.axhline(y=bounds.delta_r, color='r', linestyle='--', 
                   label=f'Δr Bound ({bounds.prob_bound_lower:.1%} confidence)', linewidth=2)
        ax1.axhline(y=bounds.forward_quantile, color='g', linestyle=':', 
                   label=f'Forward Quantile', linewidth=1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Tracking Error')
        ax1.set_title(f'Probabilistic Tracking Error Analysis - {Path(file_path).name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative statistics
        time_horizon = np.arange(1, len(tracking_errors) + 1)
        theoretical_bound = [bounds.exponential_decay_rate ** k * tracking_errors[0] + 
                           bounds.delta_r * (1 - bounds.exponential_decay_rate ** k) / (1 - bounds.exponential_decay_rate)
                           if bounds.exponential_decay_rate != 1.0 else tracking_errors[0] + bounds.delta_r * k
                           for k in time_horizon]
        
        ax2.plot(time_horizon, tracking_errors, 'b-', label='Actual', linewidth=2)
        ax2.plot(time_horizon, theoretical_bound, 'r--', label='Theoretical Bound', linewidth=2)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Tracking Error')
        ax2.set_title('Tracking Error vs Theoretical Bound')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate probabilistic tracking error analysis."""
    
    # Initialize parameters
    params = ConformalKoopmanParams(
        alpha=0.1,  # 90% confidence for forward embedding
        beta=0.1,   # 90% confidence for inverse embedding
        gamma=0.9,  # Lyapunov contraction rate
        rho=0.01,   # Robustification constant
        K=100       # Prediction horizon
    )
    
    # Create analyzer
    analyzer = ProbabilisticTrackingError(params)
    
    # Analyze Koopman CP files
    base_path = Path(__file__).parent.parent / 'data' / 'Koopman_CP'
    json_files = list(base_path.glob('*.json'))
    
    if json_files:
        print("=== Probabilistic Tracking Error Analysis ===")
        print(f"Parameters: α={params.alpha}, β={params.beta}, γ={params.gamma}")
        print(f"Confidence level: {(1-params.alpha-params.beta)*100:.1f}%")
        print()
        
        # Analyze first file as example
        test_file = json_files[0]
        print(f"Analyzing: {test_file.name}")
        
        bounds = analyzer.analyze_single_trajectory(str(test_file))
        
        print(f"Results:")
        print(f"  Forward quantile q_fwd: {bounds.forward_quantile:.6f}")
        print(f"  Inverse quantile q_inv: {bounds.inverse_quantile:.6f}")
        print(f"  High-probability bound Δr: {bounds.delta_r:.6f}")
        print(f"  Lyapunov bound: {bounds.lyapunov_bound:.6f}")
        print(f"  Probability guarantee: ≥{bounds.prob_bound_lower:.1%}")
        print(f"  Exponential decay rate: {bounds.exponential_decay_rate:.6f}")
        print()
        
        # Compare multiple files
        if len(json_files) > 1:
            print("=== Comparison across trajectories ===")
            sample_files = json_files[:3]  # Analyze first 3 files
            results = analyzer.compare_trajectories([str(f) for f in sample_files])
            
            for file_name, bounds in results.items():
                print(f"{file_name}:")
                print(f"  Δr: {bounds.delta_r:.6f}, Conf: {bounds.prob_bound_lower:.1%}")
            
        # Create visualization
        if HAS_MATPLOTLIB:
            try:
                save_path = Path(__file__).parent.parent / 'data' / 'tracking_error_analysis.png'
                analyzer.plot_tracking_error_analysis(str(test_file), str(save_path))
                print(f"Plot saved to: {save_path}")
            except Exception as e:
                print(f"Could not create plot: {e}")
        else:
            print("Matplotlib not available. Skipping plot generation.")
    else:
        print("No JSON files found in Koopman_CP directory")


if __name__ == "__main__":
    main()