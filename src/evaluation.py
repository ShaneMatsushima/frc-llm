from rouge import Rouge
import numpy as np
from sklearn.metrics import accuracy_score

class FRCModelEvaluator:
    def __init__(self, assistant, test_dataset):
        self.assistant = assistant
        self.test_data = test_dataset
        self.rouge = Rouge()
    
    def evaluate_rule_explanations(self):
        """Evaluate rule explanation accuracy"""
        correct = 0
        for item in self.test_data['rule_explanations']:
            response = self.assistant.explain_rule(item['rule_number'])
            # Simple keyword matching - could be enhanced
            if any(keyword in response.lower() for keyword in item['keywords']):
                correct += 1
        return correct / len(self.test_data['rule_explanations'])
    
    def evaluate_strategy_quality(self):
        """Evaluate strategy suggestions using ROUGE metrics"""
        references = []
        hypotheses = []
        
        for item in self.test_data['strategy_questions']:
            response = self.assistant.suggest_strategy(item['year'], item['strengths'])
            references.append(item['expert_strategy'])
            hypotheses.append(response)
        
        scores = self.rouge.get_scores(hypotheses, references, avg=True)
        return scores
    
    def run_full_evaluation(self):
        """Run all evaluation metrics"""
        results = {
            'rule_accuracy': self.evaluate_rule_explanations(),
            'strategy_rouge': self.evaluate_strategy_quality(),
        }
        return results