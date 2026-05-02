"""
detection_smoother.py — Anti-flicker layer for YOLO detections.

Problem: YOLO detects a drone in frame N, misses it in N+1, finds it again
in N+2. The box pops in/out and can cause the tracker to snap to a ghost.

Solution: keep each detection "alive" for TTL frames after its last sighting.
If a same-class detection appears nearby within that window, refresh its TTL.
Otherwise it ages out cleanly.

Works with the dict format used in tracker.py:
  {"bbox": (x,y,w,h), "cx": int, "cy": int, "conf": float,
   "label": str, "class_id": int}
"""


class DetectionSmoother:
    def __init__(self, ttl: int = 3, match_dist: int = 60):
        self._ttl = ttl
        self._match_dist2 = match_dist ** 2
        self._tracked = []   # list of {"det": dict, "ttl": int, "hits": int}

    def update(self, fresh: list) -> list:
        """
        Match fresh detections against live entries by class + proximity.
        Matched entries get their TTL reset; unmatched entries age by 1.
        Returns a flicker-free detection list ordered by stability.
        """
        used = [False] * len(fresh)

        for t in self._tracked:
            best_idx, best_dist = -1, self._match_dist2
            for i, f in enumerate(fresh):
                if used[i] or f["class_id"] != t["det"]["class_id"]:
                    continue
                dx = f["cx"] - t["det"]["cx"]
                dy = f["cy"] - t["det"]["cy"]
                d2 = dx * dx + dy * dy
                if d2 < best_dist:
                    best_dist, best_idx = d2, i

            if best_idx >= 0:
                t["det"] = fresh[best_idx]
                t["ttl"] = self._ttl
                t["hits"] += 1
                used[best_idx] = True
            else:
                t["ttl"] -= 1

        self._tracked = [t for t in self._tracked if t["ttl"] > 0]

        for i, f in enumerate(fresh):
            if not used[i]:
                self._tracked.append({"det": f, "ttl": self._ttl, "hits": 1})

        # Most stable (highest hits) first so auto-lock prefers known targets
        self._tracked.sort(key=lambda t: (-t["hits"], -t["det"]["conf"]))
        return [t["det"] for t in self._tracked]

    def stable_only(self) -> list:
        """Return only detections seen at least twice — more reliable for auto-lock."""
        return [t["det"] for t in self._tracked if t["hits"] >= 2]

    def reset(self):
        self._tracked.clear()
