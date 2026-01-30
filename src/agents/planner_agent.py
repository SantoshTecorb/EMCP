"""
LLM-Based Planner Agent
Uses QueryIntentParser for intelligent intent detection and planning
"""

import logging
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from core.models import UserRole, DataSourceType
from agents.query_parser import QueryIntentParser
from core.prompts import PromptManager

logger = logging.getLogger(__name__)

class PlannerAgent:
    """LLM-powered planner that understands natural language intent"""
    
    def __init__(self, llm_model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0.1,
            max_tokens=1000
        )
        self.intent_parser = QueryIntentParser(llm_model)
        self.prompt_manager = PromptManager()
    
    async def create_comprehensive_plan(
        self, 
        question: str, 
        user_role: UserRole
    ) -> Dict[str, Any]:
        """
        Create a comprehensive plan using LLM-based intent understanding
        """
        
        # Step 1: Parse intent using LLM
        intent, _ = await self._analyze_intent(question)
        
        # Step 2: Get accessible sources for user role
        accessible_sources = self._get_accessible_sources(user_role)
        
        # Step 3: Determine recommended sources based on intent
        recommended_sources = self._determine_sources(intent, accessible_sources)
        
        # Step 4: Create execution plan
        execution_plan = self._create_execution_plan(recommended_sources, intent, question)
        
        # Step 5: Determine task type
        task_type = self._determine_task_type(intent, question)
        
        return {
            "task_type": task_type,
            "identified_aspects": intent.entities if intent.entities else ["general_inquiry"],
            "aspect_priorities": self._calculate_aspect_priorities(intent),
            "recommended_sources": recommended_sources,
            "source_priorities": self._calculate_source_priorities(recommended_sources, intent),
            "search_strategy": self._determine_search_strategy(recommended_sources, intent),
            "reasoning": intent.reasoning if hasattr(intent, 'reasoning') else self._generate_reasoning(intent),
            "execution_plan": execution_plan
        }
    
    async def _analyze_intent(self, question: str):
        """Use LLM to deeply analyze query intent"""
        
        system_prompt = """You are an expert query analyzer. Analyze the user's question and determine:

1. **Primary Intent**: What is the user trying to achieve?
2. **Information Type**: What type of information are they seeking?
3. **Source Priority**: Which data sources would be most relevant (documentation, tickets, runbooks)?
4. **Key Entities**: Any specific entities mentioned (ticket numbers, error codes, API endpoints)?
5. **Filters**: Any specific filters or constraints?

**Context about sources:**
- documentation: Technical docs, API references, configuration guides
- tickets: Customer support tickets, incident reports, issue tracking
- runbooks: Operational procedures, deployment guides, troubleshooting steps

**Examples:**
- "ticket-022" → Primary: tickets, Entities: ["ticket-022"]
- "API rate limiting" → Primary: documentation, Entities: ["api", "rate limiting"]
- "How to deploy to production" → Primary: runbooks, Entities: ["deployment", "production"]

Return JSON with:
{
  "primary_source": "documentation" | "tickets" | "runbooks" | null,
  "entities": ["list", "of", "entities"],
  "filters": {"key": "value"},
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your analysis"
}"""
        
        return await self.intent_parser.parse(question, system_prompt)
    
    def _get_accessible_sources(self, user_role: UserRole) -> List[DataSourceType]:
        """Get sources accessible to user role"""
        # Permission logic (same as before or from PermissionManager)
        from core.permissions import PermissionManager
        permission_manager = PermissionManager()
        return [DataSourceType(source) for source in permission_manager.get_accessible_sources(user_role)]
    
    def _determine_sources(
        self, 
        intent, 
        accessible_sources: List[DataSourceType]
    ) -> List[DataSourceType]:
        """Determine which sources to search based on intent"""
        
        # Start with intent's primary source if available and confident
        if intent.primary_source and intent.confidence > 0.6:
            primary = intent.primary_source
            
            # Add secondary sources based on primary
            secondary_map = {
                DataSourceType.TICKETS: [DataSourceType.DOCUMENTATION],
                DataSourceType.DOCUMENTATION: [DataSourceType.TICKETS, DataSourceType.RUNBOOKS],
                DataSourceType.RUNBOOKS: [DataSourceType.DOCUMENTATION]
            }
            
            secondary = secondary_map.get(primary, [])
            all_suggested = [primary] + secondary
        else:
            # If intent unclear, use weighted selection
            all_suggested = self._weighted_source_selection(intent, accessible_sources)
        
        # Filter by accessibility
        return [s for s in all_suggested if s in accessible_sources]
    
    def _weighted_source_selection(self, intent, accessible_sources: List[DataSourceType]) -> List[DataSourceType]:
        """Select sources based on weighted analysis of intent"""
        weights = {
            DataSourceType.DOCUMENTATION: 0.4,  # Default weight
            DataSourceType.TICKETS: 0.3,
            DataSourceType.RUNBOOKS: 0.3
        }
        
        # Adjust weights based on intent entities
        if intent.entities:
            entities_lower = [e.lower() for e in intent.entities]
            
            # Boost documentation for technical terms
            tech_terms = ["api", "endpoint", "configuration", "authentication", "rate limit"]
            if any(term in entities_lower for term in tech_terms):
                weights[DataSourceType.DOCUMENTATION] += 0.3
            
            # Boost tickets for incident/error terms
            incident_terms = ["error", "issue", "problem", "bug", "incident"]
            if any(term in entities_lower for term in incident_terms):
                weights[DataSourceType.TICKETS] += 0.3
            
            # Boost runbooks for procedural terms
            procedural_terms = ["how to", "steps", "procedure", "deploy", "install"]
            if any(term in entities_lower for term in procedural_terms):
                weights[DataSourceType.RUNBOOKS] += 0.3
        
        # Normalize weights
        total = sum(weights.values())
        normalized = {k: v/total for k, v in weights.items()}
        
        # Sort by weight and take top 2
        sorted_sources = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return [source for source, weight in sorted_sources[:2]]
    
    def _create_execution_plan(
        self, 
        sources: List[DataSourceType], 
        intent, 
        question: str
    ) -> List[Dict[str, Any]]:
        """Create detailed execution plan"""
        
        plan = []
        
        for i, source in enumerate(sources):
            # Determine priority (first source highest)
            priority = 1.0 - (i * 0.3)
            
            # Determine focus based on source type and intent
            focus_map = {
                DataSourceType.TICKETS: self._get_ticket_focus(intent, question),
                DataSourceType.DOCUMENTATION: self._get_documentation_focus(intent, question),
                DataSourceType.RUNBOOKS: self._get_runbook_focus(intent, question),
            }
            
            focus_aspects = focus_map.get(source, ["general_information"])
            
            plan.append({
                "step": i + 1,
                "source": source.value,
                "priority": round(priority, 2),
                "focus_aspects": focus_aspects,
                "search_type": "primary" if i == 0 else "supporting",
                "query_modifiers": self._get_query_modifiers(source, intent)
            })
        
        return plan
    
    def _get_ticket_focus(self, intent, question: str) -> List[str]:
        """Determine focus aspects for tickets search"""
        focuses = []
        
        # Check for ticket ID
        if intent.entities:
            for entity in intent.entities:
                if "ticket-" in entity.lower() or any(prefix in entity.lower() for prefix in ["case-", "issue-", "tk-"]):
                    focuses.append("ticket_lookup")
                    break
        
        # Check for error/incident terms
        question_lower = question.lower()
        if any(term in question_lower for term in ["error", "issue", "problem", "incident"]):
            focuses.append("troubleshooting")
        
        if any(term in question_lower for term in ["customer", "client", "user"]):
            focuses.append("customer_issues")
        
        return focuses or ["general_ticket_search"]
    
    def _get_documentation_focus(self, intent, question: str) -> List[str]:
        """Determine focus aspects for documentation search"""
        focuses = []
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["api", "endpoint", "rest", "graphql"]):
            focuses.append("api_reference")
        
        if any(term in question_lower for term in ["config", "setup", "install"]):
            focuses.append("configuration")
        
        if any(term in question_lower for term in ["auth", "authentication", "security"]):
            focuses.append("security")
        
        return focuses or ["technical_documentation"]
    
    def _get_runbook_focus(self, intent, question: str) -> List[str]:
        """Determine focus aspects for runbooks search"""
        focuses = []
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["how to", "steps", "procedure", "process"]):
            focuses.append("procedural_guidance")
        
        if any(term in question_lower for term in ["deploy", "release", "production"]):
            focuses.append("deployment")
        
        if any(term in question_lower for term in ["troubleshoot", "debug", "fix"]):
            focuses.append("troubleshooting")
        
        return focuses or ["operational_procedures"]
    
    def _get_query_modifiers(self, source: DataSourceType, intent) -> Dict[str, Any]:
        """Get query modifiers for specific source"""
        modifiers = {}
        
        if source == DataSourceType.TICKETS and intent.entities:
            # For tickets, extract ticket IDs from entities
            ticket_ids = []
            for entity in intent.entities:
                if "ticket-" in entity.lower():
                    ticket_ids.append(entity)
            
            if ticket_ids:
                modifiers["ticket_ids"] = ticket_ids
        
        return modifiers
    
    def _determine_task_type(self, intent, question: str) -> str:
        """Determine task type from intent"""
        
        # Check intent primary source
        if intent.primary_source == DataSourceType.TICKETS:
            return "ticket_inquiry"
        elif intent.primary_source == DataSourceType.DOCUMENTATION:
            return "documentation_search"
        elif intent.primary_source == DataSourceType.RUNBOOKS:
            return "runbook_execution"
        
        # Fallback to keyword detection
        question_lower = question.lower()
        
        if "ticket-" in question_lower:
            return "ticket_inquiry"
        elif any(term in question_lower for term in ["api", "endpoint", "rest"]):
            return "api_development"
        elif any(term in question_lower for term in ["how to", "steps", "procedure"]):
            return "procedural_guidance"
        elif any(term in question_lower for term in ["error", "fix", "debug"]):
            return "troubleshooting"
        
        return "general_inquiry"
    
    def _calculate_aspect_priorities(self, intent) -> Dict[str, float]:
        """Calculate aspect priorities"""
        priorities = {}
        
        if intent.entities:
            for entity in intent.entities:
                # Simple priority based on entity type
                if "ticket-" in entity.lower():
                    priorities["ticket_lookup"] = 0.9
                elif any(term in entity.lower() for term in ["error", "issue"]):
                    priorities["troubleshooting"] = 0.8
        
        return priorities or {"general_inquiry": 0.5}
    
    def _calculate_source_priorities(self, sources: List[DataSourceType], intent) -> Dict[DataSourceType, float]:
        """Calculate source priorities"""
        priorities = {}
        
        for i, source in enumerate(sources):
            # Base priority based on position
            base_priority = 1.0 - (i * 0.3)
            
            # Adjust based on intent confidence
            if intent.primary_source == source:
                base_priority += 0.2
            
            priorities[source] = min(base_priority, 1.0)
        
        return priorities
    
    def _determine_search_strategy(self, sources: List[DataSourceType], intent) -> str:
        """Determine search strategy"""
        if len(sources) == 1:
            return "focused"
        elif intent.confidence > 0.7:
            return "comprehensive"
        else:
            return "sequential"
    
    def _generate_reasoning(self, intent) -> str:
        """Generate reasoning for the plan"""
        if intent.primary_source:
            return f"LLM analysis suggests primary source: {intent.primary_source.value} with confidence: {intent.confidence:.2f}"
        return "LLM analysis suggests general inquiry across multiple sources"