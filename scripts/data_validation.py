import pandas as pd
import logging

logging.basicConfig(filename='audit_trail.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class DataValidation:
    def __init__(self):
        pass

    def check_duplicates(self, df):
        duplicates = df.duplicated(subset=['Transaction_ID']).sum()
        logging.info(f"Found {duplicates} duplicate transactions")
        return duplicates

    def check_invalid_amounts(self, df):
        invalid = df[df['Amount'] < 0].shape[0]
        logging.info(f"Found {invalid} transactions with invalid amounts")
        return invalid

    def check_missing_approvals(self, df):
        if 'Approval_Status' in df.columns:
            missing = df[df['Approval_Status'].isna()].shape[0]
            logging.info(f"Found {missing} transactions with missing approvals")
            return missing
        return 0

    def sox_compliance_check(self, df):
        # SOX: High value transactions must be approved
        high_value = df[df['Amount'] > 10000]
        violations = high_value[high_value['Approval_Status'] != 'Approved'].shape[0]
        logging.info(f"Found {violations} SOX compliance violations")
        return violations

    def generate_validation_report(self, df):
        report = {
            'total_transactions': len(df),
            'duplicates': self.check_duplicates(df),
            'invalid_amounts': self.check_invalid_amounts(df),
            'missing_approvals': self.check_missing_approvals(df),
            'sox_violations': self.sox_compliance_check(df)
        }
        logging.info(f"Validation report generated: {report}")
        return report