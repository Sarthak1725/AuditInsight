class RiskScoring:
    def __init__(self):
        pass

    def calculate_composite_risk_score(self, anomaly_score, compliance_violations, nlp_risk_score):
        # Anomaly score: assume -1 (anomaly) to 1 (normal), map to 0-50
        anomaly_contrib = ((anomaly_score + 1) / 2) * 50
        # Compliance: number of violations, cap at 3 for 30 points
        compliance_contrib = min(compliance_violations, 3) * 10
        # NLP: 0-1, map to 0-20
        nlp_contrib = nlp_risk_score * 20
        total_score = anomaly_contrib + compliance_contrib + nlp_contrib
        return min(total_score, 100)