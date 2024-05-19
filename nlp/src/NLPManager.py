from typing import Dict

class NLPManager:
    def __init__(self):
        # install groq
        from groq import Groq
        self.groqModel = Groq(api_key="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")
        pass

    def qa(self, context: str) -> Dict[str, str]:
        prompt = f"""
            Can you summarize the following text and provide me with these 3 qualities:
            1. "TARGET" for target to be neutralized IN LOWERCASE
            2. "HEADING" for the heading of the target to be neutralized IN NUMBERS
            3. "TOOL" to be used to neutralize the target IN LOWERCASE
            Return the response strictly in a JSON format ONLY AND NO OTHER TEXT.
            The keys are the forementioned qualities.
            The text to be summarized is: {context}
            """
        results = self.groqModel.chat.completions.create(
            messages=[{"role":"user", "content":prompt}], model="llama3-8b-8192"
        )
        toReturn = results.choices[0].message.content
        
        return toReturn