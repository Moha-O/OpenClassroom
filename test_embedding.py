import unittest
import joblib

sbert_model = joblib.load('sbert_model.pkl')


class TestEmbeddingFunction(unittest.TestCase):
    
    def test_embedding(self):
        """
        Test embedding of a sentence
        """
        try:
            question = "A regular string"
            embeddings = sbert_model.encode([question])
        except Exception as e:
            # If any exception is raised, the test should fail
            self.fail(f"sbert_model.encode([question]) raised an exception: {e}")
    
    def test_emedding_sentences(self):
        """
        Test embedding of several sentences
        """
        try:
            questions = ["A regular string", "A second string"]
            embeddings = sbert_model.encode(questions)
        except Exception as e:
            # If any exception is raised, the test should fail
            self.fail(f"ssbert_model.encode(questions) raised an exception: {e}")
    
if __name__ == '__main__':
    unittest.main()
