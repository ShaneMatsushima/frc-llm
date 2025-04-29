from transformers import pipeline
import yaml

class FRCStrategyAssistant:
    def __init__(self, model_path, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=self.config.get('device', 'cpu'),
            framework="pt"
        )
    
    def generate_response(self, prompt, max_length=200, temperature=0.7):
        """Generate a strategy response to the given prompt"""
        full_prompt = f"FRC Strategy Question: {prompt}\nFRC Assistant Answer:"
        
        output = self.pipeline(
            full_prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        
        return output[0]['generated_text'].replace(full_prompt, "").strip()
    
    def explain_rule(self, rule_number):
        """Generate an explanation of a specific rule"""
        return self.generate_response(f"Explain rule {rule_number} in simple terms")
    
    def suggest_strategy(self, game_year, team_strengths):
        """Suggest a strategy based on team strengths"""
        prompt = (f"For the {game_year} FRC game, our team is strong in {team_strengths}. "
                 "What strategy would you recommend for qualification matches?")
        return self.generate_response(prompt)