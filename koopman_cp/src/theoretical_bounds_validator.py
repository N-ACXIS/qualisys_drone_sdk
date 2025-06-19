#!/usr/bin/env python3
"""
Theoretical Bounds Validator for Koopman Control

This module validates theoretical tracking error bounds by:
1. Plotting theoretical error bands around target trajectories
2. Visualizing actual trajectories against these bounds
3. Calculating empirical probabilities of staying within bounds
4. Comparing theoretical vs empirical probabilities

Addresses issue #6: feat: validate theoretical bounds for tracking error
"""

import json
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

# Import our probabilistic tracking error module
from probabilistic_tracking_error import (
    ProbabilisticTrackingError, 
    ConformalKoopmanParams,
    TrackingErrorBounds,
    HAS_MATPLOTLIB
)

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches


@dataclass
class ValidationResults:
    """Results of theoretical bounds validation."""
    file_name: str
    theoretical_probability: float  # Expected probability (1-α-β)
    empirical_probability: float   # Actual probability within bounds
    total_points: int
    points_within_bounds: int
    max_violation: float  # Maximum distance outside bounds
    mean_tracking_error: float
    std_tracking_error: float
    bounds: TrackingErrorBounds
    validation_passed: bool  # True if empirical >= theoretical (within tolerance)


class TheoreticalBoundsValidator:
    """
    Validator for theoretical tracking error bounds in Koopman control systems.
    """
    
    def __init__(self, params: ConformalKoopmanParams, tolerance: float = 0.05):
        """
        Initialize validator.
        
        Args:
            params: Conformal Koopman parameters
            tolerance: Tolerance for probability validation (empirical can be this much below theoretical)
        """
        self.params = params
        self.tolerance = tolerance
        self.analyzer = ProbabilisticTrackingError(params)
    
    def calculate_theoretical_error_band(self, targets: np.ndarray, 
                                       delta_r: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate theoretical error band around target trajectory.
        
        Args:
            targets: (N, 3) target trajectory points
            delta_r: Theoretical error bound radius
            
        Returns:
            lower_bound: (N, 3) lower bound of error band
            upper_bound: (N, 3) upper bound of error band
        """
        # For simplicity, we'll create spherical bounds around each target point
        lower_bound = targets - delta_r
        upper_bound = targets + delta_r
        
        return lower_bound, upper_bound
    
    def check_points_within_bounds(self, poses: np.ndarray, targets: np.ndarray, 
                                 delta_r: float) -> Tuple[np.ndarray, float, int]:
        """
        Check which trajectory points are within theoretical bounds.
        
        Args:
            poses: (N, 3) actual trajectory points
            targets: (N, 3) target trajectory points
            delta_r: Theoretical error bound radius
            
        Returns:
            within_bounds: (N,) boolean array indicating points within bounds
            empirical_probability: Fraction of points within bounds
            points_within: Number of points within bounds
        """
        # Calculate distances from each pose to corresponding target
        distances = np.linalg.norm(poses - targets, axis=1)
        
        # Check which points are within the theoretical bound
        within_bounds = distances <= delta_r
        points_within = np.sum(within_bounds)
        empirical_probability = points_within / len(poses) if len(poses) > 0 else 0.0
        
        return within_bounds, empirical_probability, points_within
    
    def calculate_violation_statistics(self, poses: np.ndarray, targets: np.ndarray, 
                                     delta_r: float) -> Dict[str, float]:
        """
        Calculate statistics about bound violations.
        
        Args:
            poses: (N, 3) actual trajectory points
            targets: (N, 3) target trajectory points
            delta_r: Theoretical error bound radius
            
        Returns:
            Dictionary with violation statistics
        """
        distances = np.linalg.norm(poses - targets, axis=1)
        violations = distances - delta_r
        
        stats = {
            'max_violation': np.max(violations),
            'mean_violation': np.mean(violations[violations > 0]) if np.any(violations > 0) else 0.0,
            'num_violations': np.sum(violations > 0),
            'violation_percentage': np.sum(violations > 0) / len(violations) * 100 if len(violations) > 0 else 0.0,
            'max_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        }
        
        return stats
    
    def validate_single_trajectory(self, file_path: str) -> ValidationResults:
        """
        Validate theoretical bounds for a single trajectory.
        
        Args:
            file_path: Path to trajectory JSON file
            
        Returns:
            ValidationResults object with comprehensive validation data
        """
        # Analyze trajectory and get bounds
        bounds = self.analyzer.analyze_single_trajectory(file_path)
        
        # Load and process trajectory data
        data = self.analyzer.load_trajectory_data(file_path)
        poses, targets, times = self.analyzer.extract_poses_and_targets(data)
        
        # Remove liftoff phase
        liftoff_end = self.analyzer.detect_liftoff_end(targets)
        poses = poses[liftoff_end:]
        targets = targets[liftoff_end:]
        
        # Check empirical probability
        within_bounds, empirical_prob, points_within = self.check_points_within_bounds(
            poses, targets, bounds.delta_r
        )
        
        # Calculate violation statistics
        violation_stats = self.calculate_violation_statistics(poses, targets, bounds.delta_r)
        
        # Calculate tracking error statistics
        tracking_errors = np.linalg.norm(poses - targets, axis=1)
        
        # Determine if validation passed
        theoretical_prob = bounds.prob_bound_lower
        validation_passed = empirical_prob >= (theoretical_prob - self.tolerance)
        
        return ValidationResults(
            file_name=Path(file_path).name,
            theoretical_probability=theoretical_prob,
            empirical_probability=empirical_prob,
            total_points=len(poses),
            points_within_bounds=points_within,
            max_violation=violation_stats['max_violation'],
            mean_tracking_error=np.mean(tracking_errors),
            std_tracking_error=np.std(tracking_errors),
            bounds=bounds,
            validation_passed=validation_passed
        )
    
    def plot_trajectory_with_bounds(self, file_path: str, save_path: Optional[str] = None):
        """
        Plot trajectory with theoretical error bounds.
        
        Args:
            file_path: Path to trajectory JSON file
            save_path: Optional path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping plot generation.")
            return
        
        # Get validation results
        results = self.validate_single_trajectory(file_path)
        
        # Load trajectory data
        data = self.analyzer.load_trajectory_data(file_path)
        poses, targets, times = self.analyzer.extract_poses_and_targets(data)
        
        # Remove liftoff phase
        liftoff_end = self.analyzer.detect_liftoff_end(targets)
        poses = poses[liftoff_end:]
        targets = targets[liftoff_end:]
        times = times[liftoff_end:] if len(times) > liftoff_end else np.arange(len(poses))
        
        # Check which points are within bounds
        within_bounds, _, _ = self.check_points_within_bounds(poses, targets, results.bounds.delta_r)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot target trajectory
        ax1.plot(targets[:, 0], targets[:, 1], targets[:, 2], 
                'k--', linewidth=2, label='Target Trajectory', alpha=0.7)
        
        # Plot actual trajectory with color coding
        within_poses = poses[within_bounds]
        outside_poses = poses[~within_bounds]
        
        if len(within_poses) > 0:
            ax1.scatter(within_poses[:, 0], within_poses[:, 1], within_poses[:, 2], 
                       c='green', s=20, label=f'Within Bounds ({len(within_poses)} pts)', alpha=0.8)
        
        if len(outside_poses) > 0:
            ax1.scatter(outside_poses[:, 0], outside_poses[:, 1], outside_poses[:, 2], 
                       c='red', s=20, label=f'Outside Bounds ({len(outside_poses)} pts)', alpha=0.8)
        
        # Add theoretical bound visualization (simplified as sphere at a few points)
        n_spheres = min(20, len(targets))
        sphere_indices = np.linspace(0, len(targets)-1, n_spheres, dtype=int)
        
        for i in sphere_indices[::4]:  # Show every 4th sphere to avoid clutter
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_sphere = results.bounds.delta_r * np.outer(np.cos(u), np.sin(v)) + targets[i, 0]
            y_sphere = results.bounds.delta_r * np.outer(np.sin(u), np.sin(v)) + targets[i, 1]
            z_sphere = results.bounds.delta_r * np.outer(np.ones(np.size(u)), np.cos(v)) + targets[i, 2]
            ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='blue')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectory with Theoretical Bounds\n{results.file_name}')
        ax1.legend()
        
        # 2D XY projection
        ax2 = fig.add_subplot(222)
        ax2.plot(targets[:, 0], targets[:, 1], 'k--', linewidth=2, label='Target', alpha=0.7)
        
        if len(within_poses) > 0:
            ax2.scatter(within_poses[:, 0], within_poses[:, 1], c='green', s=15, alpha=0.8)
        if len(outside_poses) > 0:
            ax2.scatter(outside_poses[:, 0], outside_poses[:, 1], c='red', s=15, alpha=0.8)
        
        # Add bound circles at several points
        for i in sphere_indices[::2]:
            circle = plt.Circle((targets[i, 0], targets[i, 1]), results.bounds.delta_r, 
                              fill=False, color='blue', alpha=0.3, linestyle=':')
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Projection with Error Bounds')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Time series of tracking errors
        ax3 = fig.add_subplot(223)
        tracking_errors = np.linalg.norm(poses - targets, axis=1)
        
        ax3.plot(times, tracking_errors, 'b-', linewidth=1.5, label='Tracking Error')
        ax3.axhline(y=results.bounds.delta_r, color='red', linestyle='--', linewidth=2, 
                   label=f'Theoretical Bound (Δr={results.bounds.delta_r:.3f})')
        ax3.axhline(y=results.mean_tracking_error, color='green', linestyle=':', linewidth=1, 
                   label=f'Mean Error ({results.mean_tracking_error:.3f})')
        
        # Highlight violations
        violations = tracking_errors > results.bounds.delta_r
        if np.any(violations):
            ax3.scatter(times[violations], tracking_errors[violations], 
                       c='red', s=20, zorder=5, alpha=0.8)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Tracking Error (m)')
        ax3.set_title('Tracking Error vs Theoretical Bound')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Statistics summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats_text = f"""
Validation Results for {results.file_name}

Theoretical Probability: {results.theoretical_probability:.1%}
Empirical Probability: {results.empirical_probability:.1%}
Validation: {'✅ PASSED' if results.validation_passed else '❌ FAILED'}

Points within bounds: {results.points_within_bounds}/{results.total_points}
Maximum violation: {results.max_violation:.4f} m
Mean tracking error: {results.mean_tracking_error:.4f} ± {results.std_tracking_error:.4f} m

Theoretical bounds (Conformal Koopman):
- Forward quantile: {results.bounds.forward_quantile:.4f}
- Inverse quantile: {results.bounds.inverse_quantile:.4f}
- Δr bound: {results.bounds.delta_r:.4f} m
- Lyapunov bound: {results.bounds.lyapunov_bound:.4f} m

Parameters: α={self.params.alpha}, β={self.params.beta}, γ={self.params.gamma}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Theoretical Bounds Validation: {results.file_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation plot saved to: {save_path}")
        
        plt.show()
    
    def validate_multiple_trajectories(self, file_paths: List[str]) -> List[ValidationResults]:
        """
        Validate theoretical bounds for multiple trajectories.
        
        Args:
            file_paths: List of paths to trajectory JSON files
            
        Returns:
            List of ValidationResults
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.validate_single_trajectory(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error validating {Path(file_path).name}: {e}")
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResults]) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            results: List of ValidationResults
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No validation results to report."
        
        passed_count = sum(1 for r in results if r.validation_passed)
        total_count = len(results)
        
        report = f"""
=== THEORETICAL BOUNDS VALIDATION REPORT ===

Parameters: α={self.params.alpha}, β={self.params.beta}, γ={self.params.gamma}
Expected confidence level: {self.params.alpha + self.params.beta:.1%}
Tolerance: ±{self.tolerance:.1%}

Overall Results: {passed_count}/{total_count} trajectories passed validation ({passed_count/total_count:.1%})

Detailed Results:
"""
        
        for result in results:
            status = "✅ PASS" if result.validation_passed else "❌ FAIL"
            diff = result.empirical_probability - result.theoretical_probability
            
            report += f"""
{result.file_name}:
  Status: {status}
  Theoretical: {result.theoretical_probability:.1%}
  Empirical: {result.empirical_probability:.1%} (diff: {diff:+.1%})
  Points within bounds: {result.points_within_bounds}/{result.total_points}
  Max violation: {result.max_violation:.4f} m
  Mean error: {result.mean_tracking_error:.4f} ± {result.std_tracking_error:.4f} m
  Δr bound: {result.bounds.delta_r:.4f} m
"""
        
        # Summary statistics
        empirical_probs = [r.empirical_probability for r in results]
        theoretical_probs = [r.theoretical_probability for r in results]
        
        report += f"""
Summary Statistics:
  Mean empirical probability: {np.mean(empirical_probs):.1%}
  Std empirical probability: {np.std(empirical_probs):.1%}
  Mean theoretical probability: {np.mean(theoretical_probs):.1%}
  
  Mean tracking error: {np.mean([r.mean_tracking_error for r in results]):.4f} m
  Max violation across all: {np.max([r.max_violation for r in results]):.4f} m
  
Validation Summary:
  - Theoretical bounds are {'WELL-CALIBRATED' if passed_count/total_count >= 0.8 else 'POORLY-CALIBRATED'}
  - Conformal prediction framework {'VALIDATED' if passed_count/total_count >= 0.9 else 'NEEDS ADJUSTMENT'}
"""
        
        return report


def main():
    """Main function to demonstrate theoretical bounds validation."""
    
    # Initialize parameters
    params = ConformalKoopmanParams(
        alpha=0.1,  # 10% for forward embedding
        beta=0.1,   # 10% for inverse embedding  
        gamma=0.9,  # Lyapunov contraction rate
        rho=0.01,   # Robustification constant
        K=50        # Prediction horizon
    )
    
    # Create validator
    validator = TheoreticalBoundsValidator(params, tolerance=0.05)
    
    # Find trajectory files
    base_path = Path(__file__).parent.parent / 'data' / 'Koopman_CP'
    json_files = list(base_path.glob('*.json'))
    
    if json_files:
        print("=== THEORETICAL BOUNDS VALIDATION ===")
        print(f"Validating {len(json_files)} trajectories...")
        print(f"Expected confidence: {(1-params.alpha-params.beta)*100:.1f}%")
        print()
        
        # Validate all trajectories
        results = validator.validate_multiple_trajectories([str(f) for f in json_files])
        
        # Generate and print report
        report = validator.generate_validation_report(results)
        print(report)
        
        # Create detailed plot for first trajectory
        if results and HAS_MATPLOTLIB:
            test_file = json_files[0]
            save_path = Path(__file__).parent.parent / 'data' / f'bounds_validation_{test_file.stem}.png'
            validator.plot_trajectory_with_bounds(str(test_file), str(save_path))
        
        # Save validation report
        report_path = Path(__file__).parent.parent / 'data' / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nValidation report saved to: {report_path}")
        
    else:
        print("No JSON files found in Koopman_CP directory")


if __name__ == "__main__":
    main()