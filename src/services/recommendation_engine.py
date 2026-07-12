class RecommendationEngine:
    """
    Knowledge-base Recommendation and Alert Layer
    """
    def __init__(self, thresholds=None):
        self.thresholds = {
            "critical_probability": 0.80,
            "warning_probability": 0.60,
            "cpu_high": 80.0,
            "memory_high": 85.0,
            "request_high": 1000.0,
            "latency_high_ms": 100.0,
            "error_high_percent": 5.0,
        }
        if thresholds:
            self.thresholds.update(thresholds)
        
    def evaluate(self, current_metrics, predictions, anomaly_flags):
        """
        current_metrics: dict containing CPU_usage, Memory_usage, Request_rate, Response_time, Error_rate
        predictions: dict containing future probability or future values
        anomaly_flags: dict containing is_high_conf_anomaly
        """
        cpu = current_metrics.get('CPU_usage', 0)
        mem = current_metrics.get('Memory_usage', 0)
        req = current_metrics.get('Request_rate', 0)
        lat = current_metrics.get('Response_time', 0)
        err = current_metrics.get('Error_rate', 0)
        
        congestion_prob = predictions.get('Congestion_probability', 0)
        is_anomaly = anomaly_flags.get('is_high_conf_anomaly', False)
        
        # 1. Alert Level Logic
        alert_level = "Normal"
        if congestion_prob > self.thresholds["critical_probability"] or is_anomaly:
            alert_level = "Critical"
        elif congestion_prob > self.thresholds["warning_probability"]:
            alert_level = "Warning"
            
        # 2. Recommendation Knowledge Base Logic
        recommendations = []
        inference = "System operating normally."
        
        rule_hits = []

        # Pattern 1: Traffic Congestion
        # Condition: CPU high, Requests increasing, Latency rising
        if cpu > self.thresholds["cpu_high"] and req > self.thresholds["request_high"] and lat > self.thresholds["latency_high_ms"]:
            rule_hits.append("traffic_saturation")
            inference = "Traffic congestion detected due to high load."
            recommendations.extend([
                "Horizontal scaling: Increase replica count.",
                "Enable/Adjust Load Balancing.",
                "Enable request caching for static assets."
            ])
            
        # Pattern 2: Memory Leak / Resource exhaustion
        # Condition: Memory high, Error rate rising
        elif mem > self.thresholds["memory_high"] and err > self.thresholds["error_high_percent"]:
            rule_hits.append("memory_or_error_saturation")
            inference = "Memory leak possibility or insufficient memory allocation."
            recommendations.extend([
                "Restart service instances to flush memory.",
                "Increase memory allocation limit (Vertical scaling).",
                "Investigate application heap dumps."
            ])
            
        # Other simple rules
        elif is_anomaly and alert_level == "Critical":
            rule_hits.append("critical_anomaly")
            inference = "Unknown critical anomaly detected."
            recommendations.append("Investigate system logs immediately.")
            
        if not recommendations:
            recommendations.append("No immediate action required.")
            
        return {
            "Alert_Level": alert_level,
            "Inference": inference,
            "Recommendations": recommendations,
            "Rule_Hits": rule_hits or ["none"],
            "Thresholds": self.thresholds,
        }

if __name__ == "__main__":
    # Test rules
    engine = RecommendationEngine()
    current = {'CPU_usage': 90, 'Memory_usage': 60, 'Request_rate': 1500, 'Response_time': 200, 'Error_rate': 0}
    preds = {'Congestion_probability': 0.85}
    flags = {'is_high_conf_anomaly': True}
    
    result = engine.evaluate(current, preds, flags)
    print(result)
