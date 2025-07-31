#!/usr/bin/env python3
"""
Summary Templates Module for Multi-Category Transcript Analysis
Provides category-specific summary templates optimized for different content types.
"""

from typing import Dict, Any
from datetime import datetime, timedelta
from content_classifier import ContentCategory

class SummaryTemplateEngine:
    """Engine for generating category-specific summary prompts and formatting"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[ContentCategory, Dict[str, str]]:
        """Initialize all category-specific templates"""
        today = datetime.today().strftime('%Y-%m-%d')
        
        return {
            # Professional Categories
            ContentCategory.TECHNICAL_MEETING: {
                'prompt': f"""You are a technical project manager. Analyze this technical meeting transcript and create a comprehensive summary focused on actionable technical outcomes.

Structure your response as follows:

## Technical Meeting Summary

### Key Technical Decisions
- List all technical decisions made with their rationale
- Include architecture choices, technology selections, implementation approaches

### Action Items & Implementation Tasks
- Break down all technical tasks that need to be completed
- Include specific technical requirements and acceptance criteria
- Assign priority levels (High/Medium/Low) based on discussion context
- Estimate effort or complexity when mentioned

### JIRA Tickets (Ready to Create)
For each actionable technical task, format as:
**[TICKET-ID]**: [Brief Title]
- **Description**: Detailed technical requirements
- **Acceptance Criteria**: 
  - [ ] Specific deliverable 1
  - [ ] Specific deliverable 2
- **Technical Notes**: Implementation details, constraints, dependencies
- **Priority**: High/Medium/Low
- **Estimated Story Points**: [if discussed]

### Technical Documentation Notes
- Code snippets, configurations, or technical details mentioned
- Architecture diagrams or system design notes needed
- Documentation updates required

### Follow-up Technical Discussions
- Unresolved technical questions requiring further investigation
- Technical research needed
- People to consult for technical expertise

### Dependencies & Blockers
- Technical dependencies that could block progress
- External systems or teams that need coordination
- Technical debt or legacy issues to address

Remember: Focus on concrete technical outcomes and actionable next steps.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_technical_context(summary)
            },
            
            ContentCategory.PROJECT_PLANNING: {
                'prompt': f"""You are a project manager analyzing a project planning session. Create a comprehensive project-focused summary.

Structure your response as follows:

## Project Planning Summary

### Project Overview & Scope
- High-level project goals and objectives discussed
- Scope boundaries and key deliverables
- Success criteria and project definition of done

### Timeline & Milestones
- Key project milestones with target dates
- Critical path dependencies
- Sprint/iteration planning details
- Use today's date ({today}) as reference for timeline estimation

### Resource Allocation & Team Structure
- Team members assigned to different work streams
- Skill requirements and capacity planning
- External dependencies and vendor coordination

### Action Items by Phase
Organize tasks into logical project phases:
- **Planning Phase**:
  - [ ] Task description (Owner: [Name], Due: [Date])
- **Development Phase**:
  - [ ] Task description (Owner: [Name], Due: [Date])
- **Testing Phase**:
  - [ ] Task description (Owner: [Name], Due: [Date])
- **Deployment Phase**:
  - [ ] Task description (Owner: [Name], Due: [Date])

### Risk Assessment & Mitigation
- Project risks identified during discussion
- Mitigation strategies and contingency plans
- Dependencies that could impact timeline

### Communication Plan
- Stakeholder update schedule and format
- Key decision points requiring approval
- Reporting and status update requirements

### Next Steps & Immediate Actions
- Immediate next steps for the next 1-2 weeks
- Upcoming meetings or decision points
- Information gathering or research needed

Focus on project organization, timeline management, and team coordination.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_project_timeline(summary)
            },
            
            ContentCategory.LEARNING_CONTENT: {
                'prompt': f"""You are a learning specialist analyzing educational content. Create a summary optimized for knowledge retention and practical application.

Structure your response as follows:

## Learning Content Summary

### Core Concepts & Key Ideas
- Main topics and concepts covered
- Fundamental principles or theories explained
- Important definitions or terminology

### Practical Techniques & Methods
- Step-by-step processes or methodologies
- Best practices and recommended approaches
- Common patterns or frameworks discussed

### Implementation Examples
- Specific examples or case studies mentioned
- Code snippets, formulas, or practical demonstrations
- Real-world applications and use cases

### Action Items for Learning
- **Immediate Practice** (next 24-48 hours):
  - [ ] Specific exercises or experiments to try
  - [ ] Concepts to research further
- **This Week**:
  - [ ] Deeper exploration topics
  - [ ] Related skills to develop
- **Ongoing Development**:
  - [ ] Long-term learning goals
  - [ ] Advanced topics to explore later

### Tools & Resources to Explore
- Software tools, libraries, or platforms mentioned
- Recommended reading or additional resources
- Communities, forums, or expert contacts to follow up with

### Key Insights & Takeaways
- Most important insights that could impact current work
- Paradigm shifts or new ways of thinking
- Connections to existing knowledge or projects

### Questions for Further Investigation
- Unclear concepts that need more research
- Advanced topics touched on but not fully explored
- Potential applications to consider

### Implementation Planning
- How to apply these concepts to current projects
- Skills that complement this learning
- Next learning steps in this domain

Focus on actionable learning outcomes and knowledge application.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_learning_context(summary)
            },
            
            ContentCategory.STATUS_UPDATE: {
                'prompt': f"""You are analyzing a status update or progress report. Create a summary focused on progress tracking and next steps.

Structure your response as follows:

## Status Update Summary

### Progress Accomplished
- **Completed Tasks**: Specific accomplishments and deliverables finished
- **Milestones Reached**: Key project milestones or goals achieved
- **Quantifiable Results**: Metrics, numbers, or measurable outcomes

### Current Work in Progress
- **Active Tasks**: What's currently being worked on
- **Expected Completion**: Timeline for current tasks
- **Resource Utilization**: Team capacity and allocation

### Obstacles & Blockers
- **Current Blockers**: Issues preventing progress
- **Resource Constraints**: Capacity, budget, or skill limitations  
- **External Dependencies**: Waiting on other teams, approvals, or vendors
- **Risk Factors**: Potential issues that could impact future progress

### Immediate Next Steps (Next 1-2 Weeks)
- [ ] Priority 1 tasks with deadlines
- [ ] Priority 2 tasks with deadlines
- [ ] Dependencies to resolve

### Longer-term Planning (Next Month)
- [ ] Major deliverables coming up
- [ ] Strategic initiatives to begin
- [ ] Resource planning needs

### Support & Resources Needed
- **Help Required**: Specific assistance needed from others
- **Resource Requests**: Tools, budget, or personnel needs
- **Decision Points**: Items requiring management approval

### Communication & Follow-up
- **Stakeholder Updates**: Who needs to be informed of progress
- **Status Reports**: Required reporting or documentation
- **Meeting Schedule**: Upcoming reviews or check-ins

### Achievements to Highlight
- Notable successes worth celebrating or sharing
- Learning experiences or process improvements
- Team accomplishments and individual contributions

Focus on progress tracking, obstacle resolution, and forward momentum.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_status_timeline(summary)
            },
            
            ContentCategory.RESEARCH_DISCUSSION: {
                'prompt': f"""You are a research coordinator analyzing a research discussion. Create a summary focused on research planning and knowledge discovery.

Structure your response as follows:

## Research Discussion Summary

### Research Questions & Hypotheses
- Primary research questions being investigated
- Hypotheses or theories being tested
- Assumptions and underlying premises

### Current Findings & Insights
- Key discoveries or patterns identified
- Data insights and analytical results
- Preliminary conclusions or trends

### Research Methodology & Approach
- Research methods and techniques being used
- Data sources and collection strategies
- Analysis frameworks and tools

### Experiments & Investigations to Conduct
- **Immediate Research Tasks** (next 1-2 weeks):
  - [ ] Specific experiments or studies to run
  - [ ] Data to collect or analyze
  - [ ] Tools or methods to test
- **Medium-term Research** (next month):
  - [ ] Larger studies or comprehensive analysis
  - [ ] Collaboration opportunities to pursue
- **Long-term Research Direction**:
  - [ ] Strategic research initiatives
  - [ ] Grant applications or funding opportunities

### Literature Review & Background Research
- Papers, studies, or publications to review
- Expert sources and thought leaders to follow
- Academic or industry research to investigate
- Knowledge gaps to explore

### Collaboration & Expert Consultation
- Researchers or experts to connect with
- Academic institutions or research groups to contact
- Industry practitioners with relevant experience
- Peer review or feedback opportunities

### Research Infrastructure & Resources
- Tools, software, or platforms needed
- Data access or acquisition requirements
- Computing resources or lab facilities
- Budget or funding considerations

### Knowledge Synthesis & Documentation
- Research documentation to create or update
- Knowledge base contributions needed
- Research presentation or publication plans
- Internal knowledge sharing opportunities

### Next Research Milestones
- Research deliverables and timelines
- Review points and progress check-ins
- Publication or presentation deadlines

Focus on advancing knowledge discovery and research progress.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_research_context(summary)
            },
            
            ContentCategory.STAKEHOLDER_COMMUNICATION: {
                'prompt': f"""You are analyzing stakeholder communication. Create a summary focused on relationship management and communication follow-up.

Structure your response as follows:

## Stakeholder Communication Summary

### Key Messages & Information Shared
- Primary information communicated to stakeholders
- Updates provided on project status or business matters
- Decisions or announcements made

### Stakeholder Feedback & Concerns
- Questions raised by stakeholders
- Concerns or issues expressed
- Feedback on proposals or current direction
- Expectations or requirements clarified

### Decisions & Approvals
- Decisions made during the discussion
- Approvals granted or requested
- Next steps agreed upon
- Resource or budget commitments

### Communication Action Items
- **Immediate Follow-up** (next 24-48 hours):
  - [ ] Information to send or reports to prepare
  - [ ] People to contact or update
  - [ ] Meetings to schedule
- **This Week**:
  - [ ] Detailed documentation to create
  - [ ] Presentations or materials to prepare
  - [ ] Stakeholder meetings to arrange

### Relationship Management Notes
- Key stakeholder priorities and concerns
- Important context about stakeholder preferences
- Relationship dynamics to be aware of
- Trust-building or rapport-building opportunities

### Business Impact & Implications
- Business consequences of decisions made
- Impact on timelines, budgets, or resources
- Strategic implications for the organization
- Risk factors to monitor and manage

### Executive Summary for Leadership
- High-level summary suitable for executive briefing
- Key points requiring leadership attention
- Strategic decisions or guidance needed
- Escalation items or concerns

### Next Stakeholder Interactions
- Upcoming meetings or presentations
- Regular communication cadence to maintain
- Stakeholder-specific follow-up needs
- Relationship maintenance activities

### Documentation & Reporting Requirements
- Meeting minutes or summary documentation needed
- Status reports or dashboards to update
- Compliance or regulatory reporting requirements
- Audit trail or decision documentation

Focus on maintaining strong stakeholder relationships and clear communication.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_stakeholder_context(summary)
            },
            
            ContentCategory.TROUBLESHOOTING_SESSION: {
                'prompt': f"""You are analyzing a troubleshooting or problem-solving session. Create a summary focused on issue resolution and prevention.

Structure your response as follows:

## Troubleshooting Session Summary

### Problem Description & Symptoms
- Clear description of the issue or failure
- Symptoms observed and their impact
- Timeline of when the problem was discovered
- Affected systems, users, or processes

### Root Cause Analysis
- Investigation methods used
- Root causes identified or suspected
- Contributing factors and conditions
- System or process failures involved

### Solution & Resolution Steps
- **Immediate Fix** (emergency resolution):
  - [ ] Critical steps taken to restore service
  - [ ] Temporary workarounds implemented
- **Permanent Solution**:
  - [ ] Long-term fixes to implement
  - [ ] System changes or improvements needed
  - [ ] Process modifications required

### Testing & Verification
- [ ] Tests needed to verify the fix works
- [ ] Monitoring to ensure problem doesn't recur
- [ ] Performance validation requirements
- [ ] User acceptance testing needed

### Prevention Measures
- **Process Improvements**:
  - [ ] Process changes to prevent recurrence
  - [ ] Training or documentation updates needed
  - [ ] Quality checks or validation steps to add
- **Technical Improvements**:
  - [ ] System monitoring enhancements
  - [ ] Automated alerts or checks to implement
  - [ ] Infrastructure or architecture improvements
- **Documentation Updates**:
  - [ ] Runbooks or procedures to update
  - [ ] Knowledge base articles to create
  - [ ] Troubleshooting guides to improve

### Lessons Learned
- Key insights from the troubleshooting process
- What worked well in the response
- What could be improved for future incidents
- Knowledge gained about system behavior

### Follow-up Actions
- [ ] Post-incident review or retrospective meeting
- [ ] Communication to affected stakeholders
- [ ] Incident documentation and reporting
- [ ] Process or system improvements to implement

### Monitoring & Alerting
- Metrics to monitor for early detection
- Alert thresholds or conditions to set
- Dashboard or reporting improvements needed
- Regular health checks to establish

Focus on comprehensive problem resolution and prevention of future occurrences.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_troubleshooting_context(summary)
            },
            
            ContentCategory.KNOWLEDGE_SHARING: {
                'prompt': f"""You are analyzing a knowledge sharing session. Create a summary focused on knowledge transfer and documentation.

Structure your response as follows:

## Knowledge Sharing Summary

### Knowledge Areas Covered
- Primary topics and subject areas discussed
- Specific expertise or experience shared
- Technical concepts or business processes explained
- Best practices and lessons learned communicated

### Key Insights & Expertise Shared
- Critical knowledge that was transferred
- Expert tips, tricks, or advanced techniques
- Common pitfalls and how to avoid them
- Proven strategies and successful approaches

### Documentation & Knowledge Capture
- **Immediate Documentation** (next 24-48 hours):
  - [ ] Key points to document in knowledge base
  - [ ] Process documentation to create or update
  - [ ] FAQ items to add based on questions asked
- **Comprehensive Documentation**:
  - [ ] Detailed guides or tutorials to create
  - [ ] Training materials to develop
  - [ ] Reference documentation to update

### Training & Onboarding Implications
- Knowledge that should be part of team onboarding
- Training programs that need updates
- Skills that new team members should develop
- Mentoring or coaching opportunities identified

### Knowledge Distribution
- **Team Communication**:
  - [ ] Team members who need this knowledge
  - [ ] Team meetings or forums to share insights
  - [ ] Informal knowledge sharing opportunities
- **Organizational Sharing**:
  - [ ] Other teams who could benefit from this knowledge
  - [ ] Company-wide presentations or demos
  - [ ] Communities of practice to engage

### Action Items for Knowledge Management
- [ ] Wiki or knowledge base updates needed
- [ ] Process documentation to standardize
- [ ] Training sessions or workshops to organize
- [ ] Expert interview or documentation sessions to schedule

### Tools & Resources for Knowledge Sharing
- Platforms or tools for better knowledge sharing
- Templates or formats that would help capture knowledge
- Communities or forums for ongoing knowledge exchange
- Technology solutions to improve knowledge accessibility

### Follow-up Knowledge Needs
- Areas where additional knowledge sharing is needed
- Experts who should be engaged for future sessions
- Topics that need deeper exploration
- Knowledge gaps that were identified

### Best Practices for Knowledge Transfer
- Effective methods identified for sharing knowledge
- Communication approaches that worked well
- Documentation formats that are most useful
- Timing and frequency for knowledge sharing

Focus on effective knowledge capture, transfer, and organizational learning.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_knowledge_context(summary)
            },
            
            # Personal Categories
            ContentCategory.PERSONAL_REFLECTION: {
                'prompt': f"""You are analyzing personal reflection content. Create a summary focused on personal growth and self-development.

Structure your response as follows:

## Personal Reflection Summary

### Key Personal Insights
- Important realizations or self-discoveries
- Patterns in thoughts, behaviors, or emotions
- Values clarification or priority setting
- Mental health or wellness observations

### Personal Growth Areas
- Strengths identified or celebrated
- Areas for improvement or development
- Skills or capabilities to build
- Habits or behaviors to change

### Goals & Intentions
- **Short-term Goals** (next 1-2 weeks):
  - [ ] Immediate personal actions or changes
  - [ ] Habits to start or stop
  - [ ] Daily practices to implement
- **Medium-term Goals** (next 1-3 months):
  - [ ] Personal development objectives
  - [ ] Skill building or learning goals
  - [ ] Relationship or social goals
- **Long-term Aspirations**:
  - [ ] Life goals or major objectives
  - [ ] Career or personal fulfillment targets
  - [ ] Values-based lifestyle changes

### Self-Care & Wellness Planning
- Mental health practices to prioritize
- Physical wellness or fitness goals
- Stress management techniques to try
- Work-life balance improvements needed

### Habit Tracking & Behavior Change
- [ ] Positive habits to develop or maintain
- [ ] Negative patterns to address or eliminate
- [ ] Environmental changes to support growth
- [ ] Accountability systems to establish

### Relationship & Social Insights
- Relationship patterns or dynamics observed
- Social connections to nurture or develop
- Communication skills to improve
- Boundary setting or maintenance needs

### Learning & Development
- Personal interests to explore further
- Skills for personal fulfillment (not work-related)
- Creative outlets or hobbies to pursue
- Educational or growth opportunities

### Gratitude & Positive Recognition
- Achievements or progress to celebrate
- People or experiences to appreciate
- Positive changes or improvements noticed
- Sources of joy or fulfillment identified

Focus on authentic personal growth and meaningful self-development.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_personal_context(summary)
            },
            
            ContentCategory.LIFE_PLANNING: {
                'prompt': f"""You are analyzing life planning discussion. Create a summary focused on major life decisions and practical planning.

Structure your response as follows:

## Life Planning Summary

### Major Decisions & Life Changes
- Significant life decisions being considered
- Major changes or transitions planned
- Timeline for important life events
- Options and alternatives being evaluated

### Financial Planning & Management
- **Budget Planning**:
  - [ ] Financial goals and targets
  - [ ] Budget adjustments or spending plans
  - [ ] Savings objectives and strategies
- **Major Financial Decisions**:
  - [ ] Investment considerations or decisions
  - [ ] Major purchases (house, car, etc.)
  - [ ] Insurance or financial protection needs
- **Financial Research & Actions**:
  - [ ] Financial advisors or experts to consult
  - [ ] Financial products or services to investigate
  - [ ] Financial education or planning tools to explore

### Family & Relationship Planning
- Family goals and planning discussions
- Relationship milestones or decisions
- Parenting or family structure considerations
- Extended family coordination or support

### Career & Professional Life Planning
- Career transitions or development plans
- Work-life balance adjustments
- Professional goals that impact personal life
- Education or training that affects life planning

### Housing & Living Situation
- Housing decisions or changes planned
- Home improvement or maintenance projects
- Location or lifestyle changes considered
- Community or neighborhood planning

### Health & Wellness Life Planning
- Health goals with long-term implications
- Healthcare planning and preparation
- Wellness lifestyle changes
- Preventive health measures to implement

### Travel & Experience Planning
- **Upcoming Travel** (next 3-6 months):
  - [ ] Trips to plan and book
  - [ ] Travel research and preparation
  - [ ] Budget and logistics coordination
- **Longer-term Travel Goals**:
  - [ ] Destination or experience goals
  - [ ] Travel savings and preparation
  - [ ] Travel-related life decisions

### Action Items by Timeline
- **This Week**:
  - [ ] Immediate decisions or actions needed
  - [ ] Research or information gathering
  - [ ] Consultations or meetings to schedule
- **Next Month**:
  - [ ] Major steps or milestones
  - [ ] Financial or legal actions to take
  - [ ] Family or relationship discussions
- **Next 3-6 Months**:
  - [ ] Major life events or changes
  - [ ] Long-term planning implementation
  - [ ] Goal achievement milestones

### Resources & Support Needed
- Professional services required (legal, financial, etc.)
- Family or friend support to coordinate
- Information or education needed for decisions
- Tools or systems to help with planning

Focus on practical life organization and meaningful life goal achievement.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_life_planning_context(summary)
            },
            
            ContentCategory.SOCIAL_CONVERSATION: {
                'prompt': f"""You are analyzing social conversation content. Create a summary focused on relationship building and social coordination.

Structure your response as follows:

## Social Conversation Summary

### Social Connections & Relationships
- Key people discussed or involved in conversation
- Relationship updates or important personal news
- Social dynamics or friendship insights
- Family updates and important life events

### Social Plans & Commitments
- **Immediate Social Plans** (next 1-2 weeks):
  - [ ] Specific events, gatherings, or meetings planned
  - [ ] People to contact or coordinate with
  - [ ] Social commitments to honor
- **Upcoming Social Events** (next month):
  - [ ] Parties, celebrations, or special occasions
  - [ ] Family gatherings or friend meetups
  - [ ] Community or group activities

### Relationship Maintenance & Follow-up
- [ ] Friends or family to check in with
- [ ] Thank you notes or appreciation to express
- [ ] Invitations to extend or respond to
- [ ] Social gestures or support to provide

### Social Learning & Insights
- Insights about friendships or social dynamics
- Communication patterns or relationship observations
- Social skills or interaction lessons learned
- Community or group dynamics understood

### Entertainment & Shared Experiences
- Movies, shows, books, or entertainment discussed
- Shared interests or hobbies to explore together
- Activity recommendations or suggestions received
- Cultural events or experiences to attend

### Support & Care Coordination
- Friends or family who need support or help
- Ways to provide assistance or care
- Emotional support or encouragement needed
- Celebration or recognition opportunities

### Social Goals & Relationship Building
- Relationships to deepen or strengthen
- New social connections to pursue
- Social skills or communication to improve
- Community involvement or engagement goals

### Gift Giving & Special Occasions
- Upcoming birthdays, anniversaries, or celebrations
- Gift ideas or special gestures planned
- Holiday or seasonal planning with friends/family
- Traditions or customs to maintain or create

### Communication Follow-up
- [ ] Phone calls or messages to send
- [ ] Social media interactions or updates
- [ ] In-person visits or meetings to arrange
- [ ] Group coordination or planning to facilitate

Focus on nurturing relationships and maintaining meaningful social connections.

Transcript:
{{transcript}}""",
                
                'post_process': lambda summary: self._add_social_context(summary)
            },
            
            ContentCategory.PERSONAL_LEARNING: {
                'prompt': f"""You are analyzing personal learning content. Create a summary focused on hobby development and personal skill building.

Structure your response as follows:

## Personal Learning Summary

### Learning Topics & Skills
- Primary subjects or skills being explored
- Techniques, methods, or approaches learned
- Creative or artistic concepts discovered
- Hobby-related knowledge gained

### Practical Application & Projects
- **Immediate Practice** (next few days):
  - [ ] Specific techniques or skills to try
  - [ ] Simple projects or exercises to complete
  - [ ] Materials or tools to gather
- **Short-term Projects** (next 2-4 weeks):
  - [ ] Creative projects to start or continue
  - [ ] Skill-building challenges to undertake
  - [ ] Personal experiments to conduct
- **Long-term Learning Goals**:
  - [ ] Advanced skills to develop over time
  - [ ] Major creative projects or achievements
  - [ ] Mastery milestones to work toward

### Creative Ideas & Inspiration
- Creative concepts or artistic ideas inspired
- Design inspirations or aesthetic discoveries
- Innovative approaches or unique techniques
- Personal style or creative direction insights

### Resources & Learning Materials
- Books, tutorials, or courses to explore further
- Online resources, websites, or communities
- Tools, software, or equipment to investigate
- Local classes, workshops, or events to attend

### Skill Development Planning
- Progression pathway for skill improvement
- Prerequisites or foundational skills needed
- Practice routines or schedules to establish
- Skill assessment or progress tracking methods

### Creative Community & Sharing
- Communities or groups to join for learning support
- Social media accounts or creators to follow
- Local meetups or hobby groups to explore
- Opportunities to share work or get feedback

### Personal Projects & Challenges
- Personal creative challenges to set
- Portfolio pieces or showcase projects to create
- Skill demonstrations or achievements to pursue
- Personal milestones or creative goals

### Learning Environment & Setup
- Workspace or creative space improvements needed
- Organization systems for materials or projects
- Time management for consistent learning practice
- Environmental factors that support learning

### Integration with Daily Life
- Ways to incorporate learning into daily routine
- Balance between learning and other life priorities
- Family or social support for learning pursuits
- Scheduling and time allocation for personal learning

Focus on meaningful personal development and creative fulfillment through learning.

Transcript:
{{transcript}}"""
            }
        }
    
    def get_template(self, category: ContentCategory) -> Dict[str, Any]:
        """Get the template configuration for a specific category"""
        return self.templates.get(category, {
            'prompt': "Please provide a comprehensive summary of this transcript.",
            'post_process': lambda x: x
        })
    
    def format_prompt(self, category: ContentCategory, transcript: str) -> str:
        """Format the prompt template with the actual transcript"""
        template = self.get_template(category)
        prompt = template.get('prompt', '')
        return prompt.replace('{{transcript}}', transcript)
    
    def post_process_summary(self, category: ContentCategory, summary: str) -> str:
        """Apply post-processing to the generated summary"""
        template = self.get_template(category)
        post_processor = template.get('post_process', lambda x: x)
        return post_processor(summary)
    
    # Post-processing helper methods
    def _add_technical_context(self, summary: str) -> str:
        """Add technical context and formatting"""
        if "JIRA Tickets" in summary and "[TICKET-ID]" in summary:
            # Add note about JIRA ticket creation
            summary += "\n\n**Note**: Replace [TICKET-ID] with actual ticket numbers when creating in JIRA system."
        return summary
    
    def _add_project_timeline(self, summary: str) -> str:
        """Add project timeline context"""
        today = datetime.today()
        next_week = today + timedelta(weeks=1)
        next_month = today + timedelta(weeks=4)
        
        timeline_note = f"""

**Timeline Reference**:
- Today: {today.strftime('%Y-%m-%d (%A)')}
- Next Week: {next_week.strftime('%Y-%m-%d (%A)')}
- One Month: {next_month.strftime('%Y-%m-%d (%A)')}

Use these dates as reference for scheduling and deadline planning."""
        
        return summary + timeline_note
    
    def _add_learning_context(self, summary: str) -> str:
        """Add learning-specific context"""
        learning_note = """

**Learning Optimization Tips**:
- Schedule practice sessions within 24 hours of learning for better retention
- Create spaced repetition schedule for complex concepts
- Connect new knowledge to existing projects or interests
- Document key insights for future reference"""
        
        return summary + learning_note
    
    def _add_status_timeline(self, summary: str) -> str:
        """Add status update timeline context"""
        today = datetime.today()
        timeline_note = f"""

**Status Timeline Reference** (Today: {today.strftime('%Y-%m-%d')}):
- Review completed tasks and update project tracking systems
- Communicate blockers to relevant stakeholders immediately
- Schedule next status update or check-in meeting
- Update personal or team dashboard with current status"""
        
        return summary + timeline_note
    
    def _add_research_context(self, summary: str) -> str:
        """Add research-specific context"""
        research_note = """

**Research Best Practices**:
- Document research methodology and assumptions clearly
- Maintain research journal or log for insights
- Set up collaboration tools for team research coordination
- Create research bibliography and reference management system"""
        
        return summary + research_note
    
    def _add_stakeholder_context(self, summary: str) -> str:
        """Add stakeholder communication context"""
        stakeholder_note = """

**Stakeholder Management Tips**:
- Follow up on commitments within promised timeframes
- Maintain regular communication cadence with key stakeholders
- Document decisions and rationale for future reference
- Anticipate stakeholder questions and prepare accordingly"""
        
        return summary + stakeholder_note
    
    def _add_troubleshooting_context(self, summary: str) -> str:
        """Add troubleshooting context"""
        troubleshooting_note = """

**Troubleshooting Best Practices**:
- Document all steps taken during investigation for future reference
- Create incident timeline for post-mortem analysis
- Update monitoring and alerting based on lessons learned
- Share resolution steps with team to prevent similar issues"""
        
        return summary + troubleshooting_note
    
    def _add_knowledge_context(self, summary: str) -> str:
        """Add knowledge sharing context"""
        knowledge_note = """

**Knowledge Management Tips**:
- Schedule follow-up sessions for complex topics
- Create searchable documentation with appropriate tags
- Establish feedback loops to improve knowledge transfer
- Consider different learning styles when sharing knowledge"""
        
        return summary + knowledge_note
    
    def _add_personal_context(self, summary: str) -> str:
        """Add personal reflection context"""
        personal_note = """

**Personal Development Tips**:
- Review and update goals regularly (weekly or monthly)
- Track progress on habit changes and personal growth
- Celebrate small wins and progress milestones
- Be patient and compassionate with yourself during growth process"""
        
        return summary + personal_note
    
    def _add_life_planning_context(self, summary: str) -> str:
        """Add life planning context"""
        life_note = """

**Life Planning Tips**:
- Review and adjust plans regularly as circumstances change
- Involve important people in your life in planning discussions
- Balance short-term needs with long-term goals
- Create contingency plans for major life decisions"""
        
        return summary + life_note
    
    def _add_social_context(self, summary: str) -> str:
        """Add social conversation context"""
        social_note = """

**Social Relationship Tips**:
- Follow through on social commitments and plans
- Remember important details shared by friends and family
- Express appreciation and gratitude regularly
- Balance giving and receiving in relationships"""
        
        return summary + social_note