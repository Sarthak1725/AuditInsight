import spacy

class NLPClassification:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.suspicious_keywords = ['urgent', 'wire transfer', 'overseas', 'unusual', 'high risk', 'fraud', 'suspicious']

    def analyze_description(self, text):
        doc = self.nlp(text.lower())
        keywords_found = [kw for kw in self.suspicious_keywords if kw in text.lower()]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        # Simple risk classification based on keywords
        risk_score = len(keywords_found) / len(self.suspicious_keywords) if self.suspicious_keywords else 0
        risk_category = 'High' if risk_score > 0.2 else 'Medium' if risk_score > 0 else 'Low'
        return {
            'keywords_found': keywords_found,
            'entities': entities,
            'risk_category': risk_category,
            'nlp_risk_score': risk_score
        }