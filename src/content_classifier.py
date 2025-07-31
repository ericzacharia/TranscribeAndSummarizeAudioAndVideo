#!/usr/bin/env python3
"""
Content Classification Module for Intelligent Transcription Analysis
Classifies transcripts into 12 categories with confidence scoring for multi-category output generation.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class ContentCategory(Enum):
    """Enumeration of all supported content categories"""
    # Professional Categories
    TECHNICAL_MEETING = "technical_meeting"
    PROJECT_PLANNING = "project_planning"
    LEARNING_CONTENT = "learning_content"
    STATUS_UPDATE = "status_update"
    RESEARCH_DISCUSSION = "research_discussion"
    STAKEHOLDER_COMMUNICATION = "stakeholder_communication"
    TROUBLESHOOTING_SESSION = "troubleshooting_session"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    
    # Personal Categories
    PERSONAL_REFLECTION = "personal_reflection"
    LIFE_PLANNING = "life_planning"
    SOCIAL_CONVERSATION = "social_conversation"
    PERSONAL_LEARNING = "personal_learning"

@dataclass
class ClassificationResult:
    """Result of content classification with confidence scores"""
    category: ContentCategory
    confidence: float
    key_indicators: List[str]

@dataclass
class MultiCategoryResult:
    """Complete classification results for all categories"""
    results: List[ClassificationResult]
    primary_category: ContentCategory
    above_threshold_categories: List[ContentCategory]
    
    def get_confidence(self, category: ContentCategory) -> float:
        """Get confidence score for a specific category"""
        for result in self.results:
            if result.category == category:
                return result.confidence
        return 0.0

class ContentClassifier:
    """Intelligent content classifier using pattern matching and keyword analysis"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize keyword patterns and scoring weights for each category"""
        
        # Professional category patterns
        self.patterns = {
            ContentCategory.TECHNICAL_MEETING: {
                'keywords': [
                    'api', 'database', 'architecture', 'implementation', 'code', 'system',
                    'technical', 'engineering', 'deployment', 'infrastructure', 'server',
                    'framework', 'library', 'algorithm', 'performance', 'optimization',
                    'bug', 'feature', 'integration', 'pipeline', 'docker', 'kubernetes'
                ],
                'phrases': [
                    'technical decision', 'implementation plan', 'code review',
                    'system design', 'api endpoint', 'database schema', 'architecture discussion'
                ],
                'indicators': [
                    'need to implement', 'technical debt', 'code base', 'pull request',
                    'merge conflict', 'deployment pipeline', 'load balancing'
                ]
            },
            
            ContentCategory.PROJECT_PLANNING: {
                'keywords': [
                    'sprint', 'milestone', 'deadline', 'deliverable', 'timeline', 'roadmap',
                    'scope', 'requirements', 'estimate', 'resource', 'capacity', 'priority',
                    'backlog', 'epic', 'story', 'planning', 'schedule', 'dependency'
                ],
                'phrases': [
                    'project timeline', 'sprint planning', 'resource allocation',
                    'project scope', 'delivery date', 'milestone review'
                ],
                'indicators': [
                    'due date', 'project plan', 'sprint goal', 'backlog refinement',
                    'capacity planning', 'resource constraint'
                ]
            },
            
            ContentCategory.LEARNING_CONTENT: {
                'keywords': [
                    'learn', 'tutorial', 'course', 'training', 'education', 'workshop',
                    'conference', 'presentation', 'demo', 'example', 'practice', 'skill',
                    'technique', 'method', 'approach', 'concept', 'theory', 'principle'
                ],
                'phrases': [
                    'how to', 'let me show you', 'for example', 'key concept',
                    'best practice', 'lesson learned', 'case study'
                ],
                'indicators': [
                    'explain', 'understand', 'demonstrate', 'illustrate',
                    'walk through', 'deep dive', 'overview'
                ]
            },
            
            ContentCategory.STATUS_UPDATE: {
                'keywords': [
                    'progress', 'update', 'status', 'completed', 'working on', 'blocked',
                    'achievement', 'accomplished', 'finished', 'started', 'continue',
                    'next steps', 'current', 'recent', 'yesterday', 'today', 'tomorrow'
                ],
                'phrases': [
                    'status update', 'progress report', 'what i did', 'what i am doing',
                    'next steps', 'completed tasks', 'current progress'
                ],
                'indicators': [
                    'stand up', 'daily update', 'weekly report', 'progress made',
                    'blockers', 'impediments', 'help needed'
                ]
            },
            
            ContentCategory.RESEARCH_DISCUSSION: {
                'keywords': [
                    'research', 'investigate', 'explore', 'experiment', 'hypothesis',
                    'analysis', 'study', 'findings', 'data', 'results', 'conclusion',
                    'methodology', 'approach', 'literature', 'paper', 'publication'
                ],
                'phrases': [
                    'research question', 'literature review', 'experimental design',
                    'data analysis', 'research findings', 'hypothesis testing'
                ],
                'indicators': [
                    'need to research', 'investigate further', 'preliminary results',
                    'proof of concept', 'feasibility study', 'pilot project'
                ]
            },
            
            ContentCategory.STAKEHOLDER_COMMUNICATION: {
                'keywords': [
                    'client', 'customer', 'stakeholder', 'management', 'executive',
                    'update', 'report', 'presentation', 'meeting', 'decision', 'approval',
                    'expectation', 'requirement', 'feedback', 'concern', 'issue'
                ],
                'phrases': [
                    'client meeting', 'stakeholder update', 'management review',
                    'executive briefing', 'customer feedback', 'business requirement'
                ],
                'indicators': [
                    'need approval', 'escalate to', 'communicate with', 'follow up with',
                    'stakeholder concern', 'business impact'
                ]
            },
            
            ContentCategory.TROUBLESHOOTING_SESSION: {
                'keywords': [
                    'problem', 'issue', 'bug', 'error', 'failure', 'broken', 'fix',
                    'debug', 'troubleshoot', 'investigate', 'root cause', 'solution',
                    'workaround', 'patch', 'resolve', 'incident', 'outage'
                ],
                'phrases': [
                    'root cause analysis', 'troubleshooting session', 'bug fix',
                    'error investigation', 'incident response', 'problem resolution'
                ],
                'indicators': [
                    'not working', 'failing', 'error message', 'exception',
                    'post mortem', 'lessons learned', 'prevention'
                ]
            },
            
            ContentCategory.KNOWLEDGE_SHARING: {
                'keywords': [
                    'share', 'document', 'knowledge', 'experience', 'lesson', 'insight',
                    'demo', 'presentation', 'training', 'onboarding', 'guide', 'manual',
                    'wiki', 'documentation', 'best practice', 'pattern', 'template'
                ],
                'phrases': [
                    'knowledge sharing', 'team demo', 'documentation update',
                    'training session', 'best practices', 'lessons learned'
                ],
                'indicators': [
                    'document this', 'share with team', 'create guide',
                    'update wiki', 'knowledge transfer', 'onboard new'
                ]
            }
        }
        
        # Personal category patterns
        self.patterns.update({
            ContentCategory.PERSONAL_REFLECTION: {
                'keywords': [
                    'feel', 'think', 'reflection', 'goal', 'growth', 'development',
                    'personal', 'self', 'improvement', 'habit', 'mindset', 'therapy',
                    'journal', 'introspection', 'meditation', 'wellness', 'balance'
                ],
                'phrases': [
                    'personal reflection', 'self assessment', 'personal growth',
                    'life goals', 'work life balance', 'mental health'
                ],
                'indicators': [
                    'i feel', 'i think', 'i want to', 'i need to',
                    'personal goal', 'self improvement', 'habit tracking'
                ]
            },
            
            ContentCategory.LIFE_PLANNING: {
                'keywords': [
                    'plan', 'budget', 'financial', 'vacation', 'family', 'house',
                    'wedding', 'move', 'career', 'decision', 'future', 'saving',
                    'investment', 'insurance', 'mortgage', 'retirement', 'education'
                ],
                'phrases': [
                    'life planning', 'family meeting', 'financial planning',
                    'vacation planning', 'major decision', 'life goals'
                ],
                'indicators': [
                    'need to plan', 'budget for', 'save money', 'make decision',
                    'family discussion', 'life change', 'major purchase'
                ]
            },
            
            ContentCategory.SOCIAL_CONVERSATION: {
                'keywords': [
                    'friend', 'family', 'social', 'party', 'dinner', 'weekend',
                    'hang out', 'catch up', 'relationship', 'dating', 'wedding',
                    'birthday', 'holiday', 'vacation', 'fun', 'entertainment'
                ],
                'phrases': [
                    'catch up', 'hang out', 'social plans', 'friend meeting',
                    'family time', 'weekend plans', 'social event'
                ],
                'indicators': [
                    'see you', 'talk soon', 'call me', 'text me',
                    'social commitment', 'friend plans', 'family gathering'
                ]
            },
            
            ContentCategory.PERSONAL_LEARNING: {
                'keywords': [
                    'hobby', 'creative', 'art', 'music', 'cooking', 'fitness',
                    'sport', 'game', 'craft', 'skill', 'interest', 'passion',
                    'practice', 'improve', 'learn', 'tutorial', 'class', 'course'
                ],
                'phrases': [
                    'personal interest', 'hobby project', 'creative project',
                    'skill development', 'personal learning', 'side project'
                ],
                'indicators': [
                    'want to learn', 'practice', 'get better at',
                    'hobby time', 'creative outlet', 'personal project'
                ]
            }
        })
        
        # Conversation structure patterns
        self.structure_patterns = {
            'formal_meeting': ['agenda', 'minutes', 'action items', 'follow up'],
            'casual_conversation': ['how are you', 'by the way', 'anyway', 'catch up'],
            'presentation': ['today i will', 'overview', 'conclusion', 'questions'],
            'problem_solving': ['issue', 'problem', 'solution', 'fix', 'resolve']
        }
    
    def classify_content(self, transcript: str) -> MultiCategoryResult:
        """
        Classify transcript content across all categories with confidence scoring
        
        Args:
            transcript: Raw transcript text to classify
            
        Returns:
            MultiCategoryResult with confidence scores for all categories
        """
        transcript_lower = transcript.lower()
        results = []
        
        # Analyze each category
        for category in ContentCategory:
            confidence = self._calculate_category_confidence(transcript_lower, category)
            key_indicators = self._extract_key_indicators(transcript_lower, category)
            
            results.append(ClassificationResult(
                category=category,
                confidence=confidence,
                key_indicators=key_indicators
            ))
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Determine primary category and above-threshold categories
        primary_category = results[0].category
        above_threshold = [r.category for r in results if r.confidence >= self.confidence_threshold]
        
        return MultiCategoryResult(
            results=results,
            primary_category=primary_category,
            above_threshold_categories=above_threshold
        )
    
    def _calculate_category_confidence(self, transcript: str, category: ContentCategory) -> float:
        """Calculate confidence score for a specific category"""
        patterns = self.patterns.get(category, {})
        score = 0.0
        total_words = len(transcript.split())
        
        if total_words == 0:
            return 0.0
        
        # Keyword matching (40% of score)
        keyword_score = self._score_keywords(transcript, patterns.get('keywords', []))
        score += keyword_score * 0.4
        
        # Phrase matching (35% of score)
        phrase_score = self._score_phrases(transcript, patterns.get('phrases', []))
        score += phrase_score * 0.35
        
        # Indicator matching (25% of score)
        indicator_score = self._score_indicators(transcript, patterns.get('indicators', []))
        score += indicator_score * 0.25
        
        # Apply structure bonuses
        structure_bonus = self._calculate_structure_bonus(transcript, category)
        score += structure_bonus
        
        # Normalize to 0-1 range
        return min(1.0, score)
    
    def _score_keywords(self, transcript: str, keywords: List[str]) -> float:
        """Score based on keyword frequency"""
        if not keywords:
            return 0.0
        
        total_matches = 0
        for keyword in keywords:
            matches = len(re.findall(rf'\b{re.escape(keyword)}\b', transcript))
            total_matches += matches
        
        # Normalize by transcript length and keyword list size
        words_count = len(transcript.split())
        keyword_density = total_matches / max(words_count, 1)
        
        return min(1.0, keyword_density * 100)  # Scale up for meaningful scores
    
    def _score_phrases(self, transcript: str, phrases: List[str]) -> float:
        """Score based on phrase presence"""
        if not phrases:
            return 0.0
        
        matches = 0
        for phrase in phrases:
            if phrase in transcript:
                matches += 1
        
        return matches / len(phrases)
    
    def _score_indicators(self, transcript: str, indicators: List[str]) -> float:
        """Score based on contextual indicators"""
        if not indicators:
            return 0.0
        
        matches = 0
        for indicator in indicators:
            if indicator in transcript:
                matches += 1
        
        return matches / len(indicators)
    
    def _calculate_structure_bonus(self, transcript: str, category: ContentCategory) -> float:
        """Calculate bonus score based on conversation structure patterns"""
        bonus = 0.0
        
        # Check for meeting-like structure for professional categories
        if category in [ContentCategory.TECHNICAL_MEETING, ContentCategory.PROJECT_PLANNING, 
                       ContentCategory.STAKEHOLDER_COMMUNICATION]:
            if any(pattern in transcript for pattern in self.structure_patterns['formal_meeting']):
                bonus += 0.1
        
        # Check for presentation structure for learning content
        if category == ContentCategory.LEARNING_CONTENT:
            if any(pattern in transcript for pattern in self.structure_patterns['presentation']):
                bonus += 0.15
        
        # Check for problem-solving structure for troubleshooting
        if category == ContentCategory.TROUBLESHOOTING_SESSION:
            if any(pattern in transcript for pattern in self.structure_patterns['problem_solving']):
                bonus += 0.1
        
        # Check for casual conversation markers for personal categories
        if category in [ContentCategory.SOCIAL_CONVERSATION, ContentCategory.PERSONAL_REFLECTION]:
            casual_markers = len([p for p in self.structure_patterns['casual_conversation'] if p in transcript])
            bonus += casual_markers * 0.05
        
        return min(0.2, bonus)  # Cap bonus at 20%
    
    def _extract_key_indicators(self, transcript: str, category: ContentCategory) -> List[str]:
        """Extract key indicators that led to this classification"""
        patterns = self.patterns.get(category, {})
        indicators = []
        
        # Find matching keywords
        for keyword in patterns.get('keywords', []):
            if keyword in transcript:
                indicators.append(f"keyword: {keyword}")
        
        # Find matching phrases
        for phrase in patterns.get('phrases', []):
            if phrase in transcript:
                indicators.append(f"phrase: {phrase}")
        
        # Limit to top 5 indicators
        return indicators[:5]
    
    def get_category_description(self, category: ContentCategory) -> str:
        """Get human-readable description of a category"""
        descriptions = {
            ContentCategory.TECHNICAL_MEETING: "Technical discussions, API/database planning, architecture decisions",
            ContentCategory.PROJECT_PLANNING: "Sprint planning, roadmap discussions, requirement gathering",
            ContentCategory.LEARNING_CONTENT: "Educational videos, tutorials, technical presentations, conferences",
            ContentCategory.STATUS_UPDATE: "Stand-ups, progress reports, performance reviews",
            ContentCategory.RESEARCH_DISCUSSION: "Exploring new technologies, problem-solving sessions, brainstorming",
            ContentCategory.STAKEHOLDER_COMMUNICATION: "Client calls, management updates, cross-team coordination",
            ContentCategory.TROUBLESHOOTING_SESSION: "Debugging, incident response, problem resolution",
            ContentCategory.KNOWLEDGE_SHARING: "Team presentations, demos, documentation sessions",
            ContentCategory.PERSONAL_REFLECTION: "Journal entries, self-assessment, goal setting, therapy sessions",
            ContentCategory.LIFE_PLANNING: "Family meetings, vacation planning, financial discussions, major life decisions",
            ContentCategory.SOCIAL_CONVERSATION: "Friend/family calls, social planning, relationship discussions",
            ContentCategory.PERSONAL_LEARNING: "Hobby tutorials, personal interest content, creative projects, non-work skills"
        }
        return descriptions.get(category, "Unknown category")