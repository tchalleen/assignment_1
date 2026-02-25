from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.models.email_store import EmailStore
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
        self.email_store = EmailStore()
    
    def classify_email(self, email: Email, mode: str = "topic") -> Dict[str, Any]:
        """Classify an email into topics using generated features or similar stored email"""
        if mode not in {"topic", "similar"}:
            raise ValueError("mode must be 'topic' or 'similar'")
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        if mode == "topic":
            # Original topic-based classification
            predicted_topic = self.model.predict(features)
            topic_scores = self.model.get_topic_scores(features)
        else:
            # Similar-email-based classification
            predicted_topic, topic_scores = self._classify_by_similar_email(email, features)
        
        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }

    def add_new_topic(self, topic: str, description: str) -> str:
        return self.model.add_new_topic(topic, description)

    def _classify_by_similar_email(self, email: Email, features: Dict[str, Any]) -> tuple[str, Dict[str, float]]:
        """Find the most similar stored email with a ground truth topic and use its topic"""
        candidates = self.email_store.get_emails_with_ground_truth()
        if not candidates:
            # Fallback to topic-based classification if no ground truth emails
            predicted = self.model.predict(features)
            scores = self.model.get_topic_scores(features)
            return predicted, scores

        best_topic = None
        best_score = -1.0
        for stored in candidates:
            stored_email = Email(subject=stored["subject"], body=stored["body"])
            stored_features = self.feature_factory.generate_all_features(stored_email)
            # Simple cosine similarity using the email embedding feature
            sim = self.model._calculate_topic_score(features, stored["ground_truth_topic"])
            if sim > best_score:
                best_score = sim
                best_topic = stored["ground_truth_topic"]

        # Return a dict with a single entry for the selected topic
        return best_topic, {best_topic: float(best_score)}