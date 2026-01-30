"""
Query Intent Parser
Uses LLM to extract intent and filters from natural language queries.
"""

import logging
from typing import Optional, Dict, Any
import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.models import QueryIntent, DataSourceType, UsageStats

logger = logging.getLogger(__name__)
"""
Enhanced Query Intent Parser
More robust LLM-based intent parsing with better error handling and entity extraction
"""

import logging
import json
import re
from typing import Optional, Dict, Any, Tuple, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from core.models import QueryIntent, DataSourceType, UsageStats
from core.prompts import PromptManager

logger = logging.getLogger(__name__)

# Define structured output schema using Pydantic
class IntentSchema(BaseModel):
    """Schema for structured intent parsing"""
    primary_source: Optional[str] = Field(
        None,
        description="Primary data source: 'documentation', 'tickets', or 'runbooks'"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Specific entities mentioned (ticket IDs, error codes, etc.)"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters and constraints for the search"
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for the intent analysis"
    )
    reasoning: str = Field(
        "",
        description="Brief explanation of the intent analysis"
    )
    task_type: str = Field(
        "general_inquiry",
        description="Type of task: 'ticket_lookup', 'api_documentation', 'procedural_guide', etc."
    )

class QueryIntentParser:
    """Enhanced intent parser with structured output and better error handling"""
    
    def __init__(self, llm_model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0.0,
            max_tokens=500
        )
        self.json_parser = JsonOutputParser(pydantic_object=IntentSchema)
        self.prompt_manager = PromptManager()  # 🎯 Load prompts from JSON
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        # Ticket ID patterns
        self.ticket_pattern = re.compile(
            r'(ticket[-\s]*#?\d+|case[-\s]*#?\d+|issue[-\s]*#?\d+|tk[-\s]*#?\d+|#\d+)',
            re.IGNORECASE
        )
        
        # Error code patterns
        self.error_pattern = re.compile(
            r'(\d{3}\s+error|error\s+\d{3}|exception|failure|timeout)',
            re.IGNORECASE
        )
        
        # API patterns
        self.api_pattern = re.compile(
            r'(/api/|endpoint|rest\s+api|graphql|soap|rate\s+limit|authentication)',
            re.IGNORECASE
        )
        
        # How-to patterns
        self.howto_pattern = re.compile(
            r'^(how\s+to|steps\s+to|guide\s+for|procedure\s+for)',
            re.IGNORECASE
        )
    
    async def parse(self, question: str, system_prompt_override: Optional[str] = None) -> Tuple[QueryIntent, UsageStats]:
        """
        Parse natural language question with multiple fallback strategies
        """
        # Strategy 1: Try structured parsing with Pydantic
        intent, usage = await self._parse_structured(question, system_prompt_override)
        
        if intent.confidence > 0.6:
            return intent, usage
        
        # Strategy 2: Fallback to regex-based heuristic parsing
        logger.debug("Structured parsing low confidence, using heuristic parsing")
        return self._parse_heuristic(question), UsageStats()
    
    async def _parse_structured(self, question: str, system_prompt_override: Optional[str] = None) -> Tuple[QueryIntent, UsageStats]:
        """Parse using structured output with Pydantic schema"""
        try:
            # Use default system prompt if not provided
            if not system_prompt_override:
                system_prompt_override = self._get_default_system_prompt()
            
            # Create prompt with format instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt_override),
                ("human", "Question: {question}\n\nPlease analyze the intent and return structured JSON:"),
                ("human", "Format instructions:\n{format_instructions}")
            ])
            
            # Format the prompt
            format_instructions = self.json_parser.get_format_instructions()
            messages = prompt.format_messages(
                question=question,
                format_instructions=format_instructions
            )
            
            # Invoke LLM
            response = await self.llm.ainvoke(messages)
            
            # Extract usage
            metadata = response.response_metadata if hasattr(response, 'response_metadata') else {}
            token_usage = metadata.get("token_usage", {})
            usage = UsageStats(
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0)
            )
            
            # Parse structured output
            try:
                parsed_data = self.json_parser.parse(response.content)
                intent_data = parsed_data.dict()
            except Exception as parse_error:
                logger.debug(f"Pydantic parse failed, trying direct JSON: {parse_error}")
                # Fallback to manual JSON extraction
                intent_data = self._extract_json_from_response(response.content)
            
            # Convert to QueryIntent
            intent = self._convert_to_query_intent(intent_data, question)
            
            # Boost confidence if entities were found
            if intent.entities:
                intent.confidence = min(intent.confidence + 0.2, 1.0)
            
            return intent, usage
            
        except Exception as e:
            logger.debug(f"Structured parsing failed: {e}")
            # Return low-confidence intent
            return QueryIntent(primary_source=None, confidence=0.3), UsageStats()
    
    def _parse_heuristic(self, question: str) -> QueryIntent:
        """Heuristic parsing using regex patterns and keyword matching"""
        question_lower = question.lower()
        
        # Initialize intent
        primary_source = None
        entities = []
        filters = {}
        confidence = 0.5
        
        # Extract entities using regex patterns
        entities.extend(self.ticket_pattern.findall(question))
        entities.extend(self.error_pattern.findall(question))
        
        # Determine primary source based on patterns
        if self.ticket_pattern.search(question):
            primary_source = DataSourceType.TICKETS
            confidence = 0.8
        elif self.api_pattern.search(question):
            primary_source = DataSourceType.DOCUMENTATION
            confidence = 0.7
        elif self.howto_pattern.search(question):
            primary_source = DataSourceType.RUNBOOKS
            confidence = 0.7
        
        # Keyword-based source detection (fallback)
        if not primary_source:
            source_keywords = {
                DataSourceType.TICKETS: ["ticket", "case", "issue", "incident", "customer", "support"],
                DataSourceType.DOCUMENTATION: ["api", "documentation", "guide", "reference", "config", "setup"],
                DataSourceType.RUNBOOKS: ["how to", "steps", "procedure", "deploy", "install", "troubleshoot"]
            }
            
            for source, keywords in source_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    primary_source = source
                    confidence = 0.6
                    break
        
        return QueryIntent(
            primary_source=primary_source,
            filters=filters,
            entities=entities,
            confidence=confidence
        )
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with multiple fallback strategies"""
        content = content.strip()
        
        # Strategy 1: Find JSON block with regex
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try parsing entire content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Try to fix common JSON issues
        fixed_content = self._fix_common_json_issues(content)
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Return minimal valid structure
        return {
            "primary_source": None,
            "entities": [],
            "filters": {},
            "confidence": 0.5,
            "reasoning": "JSON parsing failed, using fallback",
            "task_type": "general_inquiry"
        }
    
    def _fix_common_json_issues(self, content: str) -> str:
        """Fix common JSON issues in LLM responses"""
        fixes = [
            # Replace single quotes with double quotes
            (r"'([^']*)'", r'"\1"'),
            # Remove trailing commas
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fix unquoted keys
            (r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":'),
            # Fix True/False/null
            (r':\s*true\b', ': true'),
            (r':\s*false\b', ': false'),
            (r':\s*null\b', ': null'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _convert_to_query_intent(self, intent_data: Dict[str, Any], question: str) -> QueryIntent:
        """Convert parsed data to QueryIntent object"""
        # Extract primary source
        primary_source = None
        source_str = intent_data.get("primary_source")
        if source_str:
            try:
                primary_source = DataSourceType(source_str.lower())
            except (ValueError, AttributeError):
                primary_source = None
        
        # Extract entities, ensure they're strings
        entities = []
        raw_entities = intent_data.get("entities", [])
        for entity in raw_entities:
            if isinstance(entity, str):
                entities.append(entity)
            else:
                entities.append(str(entity))
        
        # Add regex-extracted entities if not already present
        regex_entities = self.ticket_pattern.findall(question)
        for entity in regex_entities:
            if entity not in entities:
                entities.append(entity)
        
        # Extract filters
        filters = intent_data.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        
        # Calculate confidence
        confidence = float(intent_data.get("confidence", 0.5))
        
        # 🎯 FIX: Extract and preserve task_type
        task_type = intent_data.get("task_type")
        
        # Boost confidence if we found specific entities
        if entities:
            confidence = min(confidence + 0.1, 1.0)
        
        return QueryIntent(
            primary_source=primary_source,
            filters=filters,
            entities=entities,
            confidence=confidence,
            task_type=task_type  # 🎯 ENHANCE: Pass through task_type for intelligent routing
        )
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for intent parsing from prompts.json"""
        return self.prompt_manager.get_prompt("intent_parsing", "default")