"""
kalman.py — 2D constant-velocity Kalman filter for target centroid.

Without this: the (x,y) fed to the PD controller jitters frame-to-frame
because tracker output is noisy. The turret over-corrects and looks twitchy.

With this: we fuse the noisy measurement with a motion model. We also get
a one-frame-ahead prediction for free, which compensates for system latency.

State vector: [x, y, vx, vy]
Measurement:  [x, y]
"""

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
    _HAS_FILTERPY = True
except ImportError:
    _HAS_FILTERPY = False


class CentroidKalman:
    """Constant-velocity Kalman filter for a 2D centroid."""

    def __init__(self, dt: float = 1.0 / 30.0):
        if not _HAS_FILTERPY:
            raise RuntimeError(
                "filterpy not installed. Run: pip install filterpy "
                "or set USE_KALMAN=False in tracker.py"
            )

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition: x_new = x + vx*dt, etc.
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # Measure position only, not velocity.
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Measurement noise — higher = smoother but laggier. Tune on real footage.
        self.kf.R *= 5.0

        # Process noise — higher = reacts faster to acceleration but noisier.
        self.kf.Q *= 0.1

        # Initial state covariance — large = "we don't know yet"
        self.kf.P *= 100.0

        self._initialized = False

    def update(self, x: float, y: float) -> tuple:
        """Feed a measurement, return (smoothed_x, smoothed_y)."""
        if not self._initialized:
            self.kf.x = np.array([x, y, 0, 0], dtype=float)
            self._initialized = True
        else:
            self.kf.predict()
            self.kf.update(np.array([x, y]))
        return float(self.kf.x[0]), float(self.kf.x[1])

    def predict_next(self) -> tuple:
        """Where will the target be one step ahead? Useful for lead-aim."""
        if not self._initialized:
            return None
        x_pred = self.kf.F @ self.kf.x
        return float(x_pred[0]), float(x_pred[1])

    def reset(self):
        self._initialized = False
        self.kf.P = np.eye(4) * 100.0   # restore "we don't know" uncertainty
