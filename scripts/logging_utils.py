import logging

class AuditLogger:
    def __init__(self, log_file='audit_trail.log'):
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_model_decision(self, model, transaction_id, decision):
        logging.info(f"Model {model} decision for transaction {transaction_id}: {decision}")

    def log_validation_action(self, action, details):
        logging.info(f"Validation action: {action} - {details}")

    def log_compliance_check(self, check_type, result):
        logging.info(f"Compliance check {check_type}: {result}")

    def log_risk_score(self, transaction_id, score):
        logging.info(f"Risk score for transaction {transaction_id}: {score}")