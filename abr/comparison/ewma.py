import math

SLOW = 8
FAST = 3
DEFAULT_ESTIMATE = 10

class EWMA:
    def __init__(self, half_life):
        self.alpha_ = math.exp(math.log(0.5) / half_life)
        self.estimate = 0
        self.total_weight = 0

    def sample(self, weight, value):
        adj_alpha = math.pow(self.alpha_, weight)
        self.estimate = value * (1 - adj_alpha) + adj_alpha * self.estimate
        self.total_weight += weight

    def get_total_weight(self):
        return self.total_weight

    def get_estimate(self):
        zero_factor = 1 - math.pow(self.alpha_, self.total_weight)
        return self.estimate / zero_factor


class EwmaBandwidthEstimator:
    def __init__(self, slow=SLOW, fast=FAST, default_estimate=10):
        self.default_estimate = default_estimate
        self.min_weight = 0.001
        self.min_delay_s = 0.05
        self.slow = EWMA(slow)
        self.fast = EWMA(fast)

    def sample(self, duration_s, bandwidth):
        # duration_ms = max(duration_s, self.min_delay_s)
        # bandwidth = 8000 * num_bytes / duration_ms  # bits/s
        # weight = duration_ms / 1000  # second
        weight = duration_s
        self.fast.sample(weight, bandwidth)
        self.slow.sample(weight, bandwidth)

    def can_estimate(self):
        fast = self.fast
        return fast and fast.get_total_weight() >= self.min_weight

    def get_estimate(self):
        if self.can_estimate():
            return min(self.fast.get_estimate(), self.slow.get_estimate())
        else:
            return self.default_estimate


if __name__ == "__main__":
    slow = 8
    fast = 3
    estimator = EwmaBandwidthEstimator(slow, fast, 10)
    estimator.sample(1000, 10000)
    print(estimator.get_estimate())
    estimator.sample(2000, 10000)
    print(estimator.get_estimate())
    estimator.sample(1000, 10000)
    print(estimator.get_estimate())
    estimator.sample(3000, 10000)
    print(estimator.get_estimate())
