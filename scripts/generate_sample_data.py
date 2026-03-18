import pandas as pd
import random
import datetime

def generate_sample_data(n=1000, file_path='data/sample_transactions.csv'):
    data = []
    vendors = ['VendorA', 'VendorB', 'VendorC', 'VendorD', 'VendorE']
    departments = ['Finance', 'IT', 'HR', 'Operations', 'Marketing']
    types = ['Purchase', 'Payment', 'Transfer', 'Refund']
    statuses = ['Approved', 'Pending', 'Rejected', None]
    descriptions = [
        'Regular payment for services',
        'Urgent wire transfer overseas',
        'Purchase of office supplies',
        'High risk international transaction',
        'Standard vendor payment',
        'Unusual large amount transfer',
        'Monthly subscription fee'
    ]
    for i in range(n):
        risk_flag = 1 if random.random() < 0.1 else 0  # 10% anomalies
        data.append({
            'Transaction_ID': f'TXN{i:06d}',
            'Date': (datetime.date.today() - datetime.timedelta(days=random.randint(0, 365))).isoformat(),
            'Vendor': random.choice(vendors),
            'Amount': round(random.uniform(10, 100000), 2),
            'Currency': 'USD',
            'Department': random.choice(departments),
            'Transaction_Type': random.choice(types),
            'Approval_Status': random.choice(statuses),
            'Risk_Flag': risk_flag,
            'Description': random.choice(descriptions)
        })
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Sample data generated and saved to {file_path}")
    return df

if __name__ == '__main__':
    generate_sample_data()