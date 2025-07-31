#!/usr/bin/env python3
"""
Classification Feedback and Learning System
Allows users to correct classifications and improves the system over time.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from content_classifier import ContentCategory, ContentClassifier

class ClassificationFeedback:
    """System for collecting and learning from user feedback on classifications"""
    
    def __init__(self, feedback_file: str = "classification_feedback.json"):
        self.feedback_file = Path(__file__).parent.parent / "data" / feedback_file
        self.feedback_file.parent.mkdir(exist_ok=True)
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        """Load existing feedback data"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return {"corrections": [], "user_patterns": {}, "domain_keywords": {}}
    
    def _save_feedback(self):
        """Save feedback data to file"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def collect_user_correction(self, transcript: str, predicted_categories: List[str], 
                              actual_categories: List[str], confidence_scores: Dict[str, float],
                              user_context: Optional[Dict] = None):
        """Collect user correction for improving future classifications"""
        correction = {
            "timestamp": datetime.now().isoformat(),
            "transcript_snippet": transcript[:200] + "..." if len(transcript) > 200 else transcript,
            "predicted_primary": predicted_categories[0] if predicted_categories else None,
            "predicted_all": predicted_categories,
            "actual_categories": actual_categories,
            "confidence_scores": confidence_scores,
            "user_context": user_context or {},
            "transcript_length": len(transcript.split()),
            "correction_type": self._determine_correction_type(predicted_categories, actual_categories)
        }
        
        self.feedback_data["corrections"].append(correction)
        self._update_learned_patterns(correction)
        self._save_feedback()
    
    def _determine_correction_type(self, predicted: List[str], actual: List[str]) -> str:
        """Determine what type of correction this represents"""
        if not predicted:
            return "no_detection"
        elif not actual:
            return "false_positive"
        elif predicted[0] != actual[0]:
            return "wrong_primary"
        elif set(predicted) != set(actual):
            return "wrong_secondary"
        else:
            return "correct"
    
    def _update_learned_patterns(self, correction: Dict):
        """Update learned patterns based on user correction"""
        actual_categories = correction["actual_categories"]
        user_context = correction.get("user_context", {})
        
        # Extract keywords from user context
        if "keywords" in user_context:
            keywords = user_context["keywords"].split(",")
            for category in actual_categories:
                if category not in self.feedback_data["domain_keywords"]:
                    self.feedback_data["domain_keywords"][category] = []
                
                for keyword in keywords:
                    keyword = keyword.strip().lower()
                    if keyword not in self.feedback_data["domain_keywords"][category]:
                        self.feedback_data["domain_keywords"][category].append(keyword)
        
        # Track user patterns
        correction_type = correction["correction_type"]
        if correction_type not in self.feedback_data["user_patterns"]:
            self.feedback_data["user_patterns"][correction_type] = 0
        self.feedback_data["user_patterns"][correction_type] += 1
    
    def get_learned_keywords(self, category: str) -> List[str]:
        """Get keywords learned from user feedback for a category"""
        return self.feedback_data["domain_keywords"].get(category, [])
    
    def get_correction_stats(self) -> Dict:
        """Get statistics about user corrections"""
        total_corrections = len(self.feedback_data["corrections"])
        if total_corrections == 0:
            return {"total": 0, "accuracy_before_correction": 0}
        
        correct_predictions = self.feedback_data["user_patterns"].get("correct", 0)
        accuracy = correct_predictions / total_corrections if total_corrections > 0 else 0
        
        return {
            "total_corrections": total_corrections,
            "accuracy_before_correction": accuracy,
            "correction_types": self.feedback_data["user_patterns"],
            "most_corrected_category": self._get_most_corrected_category()
        }
    
    def _get_most_corrected_category(self) -> Optional[str]:
        """Find which category gets corrected most often"""
        category_corrections = {}
        for correction in self.feedback_data["corrections"]:
            predicted = correction.get("predicted_primary")
            if predicted and correction["correction_type"] != "correct":
                category_corrections[predicted] = category_corrections.get(predicted, 0) + 1
        
        if not category_corrections:
            return None
        
        return max(category_corrections.items(), key=lambda x: x[1])[0]

class InteractiveClassifier:
    """Interactive classification system with user input"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.classifier = ContentClassifier(confidence_threshold)
        self.feedback_system = ClassificationFeedback()
    
    def classify_with_user_input(self, transcript: str, interactive: bool = False) -> Dict:
        """Classify with optional user interaction"""
        # Get initial classification
        result = self.classifier.classify_content(transcript)
        
        if not interactive:
            return self._format_result(result)
        
        # Check if classification is uncertain (low confidence or close scores)
        top_categories = sorted(result.results, key=lambda x: x.confidence, reverse=True)[:3]
        max_confidence = top_categories[0].confidence
        
        needs_clarification = (
            max_confidence < 0.6 or  # Low confidence
            len([c for c in top_categories if c.confidence > max_confidence - 0.1]) > 1  # Close scores
        )
        
        if needs_clarification:
            print("\n🤔 Classification uncertain. Let me ask for your input...")
            user_input = self._collect_user_input(top_categories)
            
            if user_input["override_category"]:
                # User provided explicit category
                corrected_result = self._apply_user_correction(result, user_input)
                self._save_user_feedback(transcript, result, user_input)
                return self._format_result(corrected_result)
        
        return self._format_result(result)
    
    def _collect_user_input(self, top_categories) -> Dict:
        """Collect user input about the classification"""
        print(f"\nDetected potential categories:")
        for i, cat_result in enumerate(top_categories, 1):
            print(f"{i}. {cat_result.category.value.replace('_', ' ').title()} ({cat_result.confidence:.1%} confidence)")
        
        user_input = {
            "override_category": None,
            "context_keywords": [],
            "participant_roles": [],
            "primary_purpose": None,
            "confidence_boost": {}
        }
        
        # Ask for category confirmation/correction
        try:
            choice = input(f"\n❓ Which category best describes this content? (1-{len(top_categories)}, or 'o' for other, 's' to skip): ").strip().lower()
            
            if choice == 's':
                return user_input
            elif choice == 'o':
                print("\nAvailable categories:")
                for i, category in enumerate(ContentCategory, 1):
                    print(f"{i:2d}. {category.value.replace('_', ' ').title()}")
                
                cat_choice = input("Enter category number: ").strip()
                if cat_choice.isdigit() and 1 <= int(cat_choice) <= len(ContentCategory):
                    categories = list(ContentCategory)
                    user_input["override_category"] = categories[int(cat_choice) - 1]
            elif choice.isdigit() and 1 <= int(choice) <= len(top_categories):
                user_input["override_category"] = top_categories[int(choice) - 1].category
        except (ValueError, KeyboardInterrupt):
            pass
        
        # Ask for context keywords
        try:
            keywords = input("\n❓ Key topics/keywords in this transcript (comma-separated, optional): ").strip()
            if keywords:
                user_input["context_keywords"] = [k.strip() for k in keywords.split(",")]
        except KeyboardInterrupt:
            pass
        
        return user_input
    
    def _apply_user_correction(self, original_result, user_input):
        """Apply user corrections to the classification result"""
        if not user_input["override_category"]:
            return original_result
        
        # Boost confidence for user-selected category
        corrected_category = user_input["override_category"]
        
        # Update the results
        for result in original_result.results:
            if result.category == corrected_category:
                result.confidence = max(0.8, result.confidence + 0.3)  # Boost confidence
                if user_input["context_keywords"]:
                    result.key_indicators.extend([f"user_keyword: {kw}" for kw in user_input["context_keywords"]])
        
        # Recalculate primary and above-threshold categories
        original_result.results.sort(key=lambda x: x.confidence, reverse=True)
        original_result.primary_category = original_result.results[0].category
        original_result.above_threshold_categories = [
            r.category for r in original_result.results 
            if r.confidence >= self.classifier.confidence_threshold
        ]
        
        return original_result
    
    def _save_user_feedback(self, transcript: str, original_result, user_input):
        """Save user feedback for learning"""
        predicted = [r.category.value for r in original_result.above_threshold_categories]
        actual = [user_input["override_category"].value] if user_input["override_category"] else []
        confidence_scores = {r.category.value: r.confidence for r in original_result.results}
        
        self.feedback_system.collect_user_correction(
            transcript, predicted, actual, confidence_scores, 
            {"keywords": ",".join(user_input.get("context_keywords", []))}
        )
    
    def _format_result(self, result):
        """Format result for output"""
        return {
            'classification': {
                'primary_category': result.primary_category.value,
                'above_threshold_categories': [cat.value for cat in result.above_threshold_categories],
                'all_confidence_scores': {r.category.value: r.confidence for r in result.results}
            },
            'metadata': {
                'feedback_stats': self.feedback_system.get_correction_stats()
            }
        }

def add_context_hints(classifier: ContentClassifier, context_hints: str):
    """Add user-provided context hints to improve classification"""
    if not context_hints:
        return
    
    # Parse context hints: "entities=data_objects,opsec=security"
    hints = {}
    for hint in context_hints.split(","):
        if "=" in hint:
            term, meaning = hint.split("=", 1)
            hints[term.strip()] = meaning.strip()
    
    # Temporarily add context-specific keywords to patterns
    for category in ContentCategory:
        patterns = classifier.patterns.get(category, {})
        if 'context_keywords' not in patterns:
            patterns['context_keywords'] = []
        
        # Add contextual meanings as keywords
        for term, meaning in hints.items():
            if any(keyword in meaning.lower() for keyword in patterns.get('keywords', [])):
                patterns['context_keywords'].append(term)