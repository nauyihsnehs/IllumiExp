import numpy as np

M1 = 0.1593017578125
M2 = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875
PQ_L_MAX = 10000.0
EPSILON = 1e-6
C2_OVER_C3 = C2 / C3


def load_quantile_cdf(path):
    with np.load(path, allow_pickle=False) as data:
        quantile_x = np.asarray(data["quantile_x"], dtype=np.float32)
        quantile_p = np.asarray(data["quantile_p"], dtype=np.float32)
        max_value = float(data["max_val"])
    return quantile_x, quantile_p, max_value


def normal_cdf(value):
    value = np.asarray(value, dtype=np.float32)
    sign = np.sign(value)
    x = np.abs(value) / np.sqrt(2.0)
    coefficients = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
    t = 1.0 / (1.0 + 0.3275911 * x)
    polynomial = (
        (
            ((coefficients[4] * t + coefficients[3]) * t + coefficients[2]) * t
            + coefficients[1]
        )
        * t
        + coefficients[0]
    ) * t
    approximation = sign * (1.0 - polynomial * np.exp(-x * x))
    return 0.5 * (1.0 + approximation)


def probit(probability):
    probability = np.clip(probability.astype(np.float32), EPSILON, 1.0 - EPSILON)
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.357751867269,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    )
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        0.007784695709041462,
        0.3224671290700398,
        2.445134137142996,
        3.754408661907416,
    )

    low = probability < 0.02425
    high = probability > 0.97575
    q_low = np.sqrt(-2.0 * np.log(probability))
    x_low = np.polyval(c, q_low) / np.polyval((*d, 1.0), q_low)
    q_mid = probability - 0.5
    radius = np.square(q_mid)
    x_mid = np.polyval(a, radius) * q_mid / np.polyval((*b, 1.0), radius)
    q_high = np.sqrt(-2.0 * np.log(1.0 - probability))
    x_high = -np.polyval(c, q_high) / np.polyval((*d, 1.0), q_high)
    return np.where(low, x_low, np.where(high, x_high, x_mid)).astype(np.float32)


def encoded_max(max_value):
    normalized = max_value / PQ_L_MAX
    powered = normalized**M1 if normalized > 0 else 0.0
    return max(((C1 + C2 * powered) / (1.0 + C3 * powered)) ** M2, EPSILON)


def pq_inverse(hdr):
    hdr = np.nan_to_num(
        np.asarray(hdr, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normalized = np.maximum(hdr, 0.0) / PQ_L_MAX
    powered = np.power(
        normalized,
        M1,
        where=normalized > 0,
        out=np.zeros_like(normalized),
    )
    return np.power(
        (C1 + C2 * powered) / (1.0 + C3 * powered),
        M2,
        dtype=np.float32,
    )


def pq_forward(encoded):
    encoded = np.maximum(np.asarray(encoded, dtype=np.float32), 0.0)
    value = np.power(encoded, 1.0 / M2, where=encoded > 0, out=np.zeros_like(encoded))
    value = np.minimum(value, C2_OVER_C3 - EPSILON)
    numerator = np.maximum(value - C1, 0.0)
    denominator = np.maximum(C2 - C3 * value, EPSILON)
    return (
        PQ_L_MAX
        * np.power(
            numerator / denominator,
            1.0 / M1,
            where=denominator > 0,
            out=np.zeros_like(denominator),
        )
    ).astype(np.float32)


def hdr_to_cdf(hdr, quantile_x, quantile_p, max_value):
    normalized = np.clip(pq_inverse(hdr) / encoded_max(max_value), 0.0, 1.0)
    probability = np.interp(normalized, quantile_x, quantile_p)
    return probit(probability)


def cdf_to_hdr(encoded, quantile_x, quantile_p, max_value):
    probability = np.clip(normal_cdf(encoded), EPSILON, 1.0 - EPSILON)
    normalized = np.interp(probability, quantile_p, quantile_x)
    return pq_forward(np.clip(normalized, 0.0, 1.0) * encoded_max(max_value))
