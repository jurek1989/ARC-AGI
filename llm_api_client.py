"""
LLM API Client for ARC AGI Solver

This module provides an interface to communicate with an external LLM API
(hosted on Google Colab) for ARC task analysis.

The module is designed to be:
- Lightweight (no local model loading)
- Network-based (communicates with external API)
- Extensible (easy to add new capabilities)
- Safe (with validation and fallbacks)
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectDescription:
    """Structured description of a grid object for LLM analysis"""
    bbox: Tuple[int, int, int, int]  # (y1, y2, x1, x2)
    area: int
    main_color: int
    color_histogram: Dict[int, int]
    shape_type: str
    num_holes: int
    touches_border: bool


@dataclass
class TaskAnalysis:
    """Complete task analysis for LLM"""
    task_id: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    input_objects: List[ObjectDescription]
    output_objects: List[ObjectDescription]
    input_grid_text: str
    output_grid_text: str


@dataclass
class LLMResponse:
    """Response from LLM API"""
    strategy: str
    confidence: float
    reasoning: str
    suggested_operations: List[str]
    new_operations: List[str]


class LLMAPIClient:
    """Client for communicating with external LLM API"""
    
    def __init__(self, api_url: str = "https://brighton-t-zen-postcard.trycloudflare.com"):
        self.api_url = api_url
        self.endpoint = f"{api_url}/generate"
        self.is_available = False
        self.timeout = 30  # seconds
        
        # Test connection on initialization
        self.is_available = self.test_connection()
        if self.is_available:
            logger.info(f"✅ LLM API connected: {self.api_url}")
        else:
            logger.warning(f"⚠️  LLM API not available: {self.api_url}")
    
    def query_llm(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and get response.
        
        Args:
            prompt: Text prompt to send to LLM
            
        Returns:
            LLM response as string
        """
        try:
            # FastAPI format: {"prompt": "text"}
            payload = {"prompt": prompt}
            
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            # Return response field
            return result["response"]
            
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return "LLM_ERROR"
    
    def analyze_task(self, task_analysis: TaskAnalysis) -> LLMResponse:
        """
        Send task analysis to LLM API and get response.
        
        Args:
            task_analysis: Complete task analysis
            
        Returns:
            LLMResponse with strategy and suggestions
        """
        if not self.is_available:
            return LLMResponse(
                strategy="api_unavailable",
                confidence=0.0,
                reasoning="LLM API not available",
                suggested_operations=[],
                new_operations=[]
            )
        
        # Create a comprehensive prompt for the LLM
        prompt = self._create_analysis_prompt(task_analysis)
        
        try:
            # Get response from LLM
            llm_response = self.query_llm(prompt)
            
            # Parse the response
            return self._parse_llm_response(llm_response, task_analysis)
            
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            return LLMResponse(
                strategy="error",
                confidence=0.0,
                reasoning=f"Analysis failed: {e}",
                suggested_operations=[],
                new_operations=[]
            )
    
    def _create_analysis_prompt(self, task_analysis: TaskAnalysis) -> str:
        """Create a comprehensive prompt for ARC task analysis"""
        
        prompt = f"""ARC Task Analysis - Task ID: {task_analysis.task_id}

INPUT GRID ({task_analysis.input_shape[0]}x{task_analysis.input_shape[1]}):
{task_analysis.input_grid_text}

OUTPUT GRID ({task_analysis.output_shape[0]}x{task_analysis.output_shape[1]}):
{task_analysis.output_grid_text}

INPUT OBJECTS ({len(task_analysis.input_objects)}):
"""
        
        for i, obj in enumerate(task_analysis.input_objects):
            prompt += f"""Object {i+1}:
- BBox: {obj.bbox}
- Area: {obj.area}
- Main Color: {obj.main_color}
- Shape Type: {obj.shape_type}
- Holes: {obj.num_holes}
- Touches Border: {obj.touches_border}
- Colors: {obj.color_histogram}

"""
        
        prompt += f"""OUTPUT OBJECTS ({len(task_analysis.output_objects)}):
"""
        
        for i, obj in enumerate(task_analysis.output_objects):
            prompt += f"""Object {i+1}:
- BBox: {obj.bbox}
- Area: {obj.area}
- Main Color: {obj.main_color}
- Shape Type: {obj.shape_type}
- Holes: {obj.num_holes}
- Touches Border: {obj.touches_border}
- Colors: {obj.color_histogram}

"""
        
        prompt += """ANALYSIS REQUEST:
Based on the input and output grids and their objects, please analyze what transformation occurred.

Please respond in the following JSON format:
{
    "strategy": "brief description of the transformation strategy",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of your analysis",
    "suggested_operations": ["list", "of", "suggested", "operations"],
    "new_operations": ["list", "of", "new", "operations", "if", "needed"]
}

Focus on:
1. What objects changed between input and output?
2. What geometric transformations occurred?
3. What color changes happened?
4. What operations would you suggest to solve this task?

Respond only with valid JSON:"""

        return prompt
    
    def _parse_llm_response(self, response: str, task_analysis: TaskAnalysis) -> LLMResponse:
        """Parse LLM response into structured format"""
        
        try:
            # Try to parse as JSON
            if response.startswith('{') and response.endswith('}'):
                parsed = json.loads(response)
                return LLMResponse(
                    strategy=parsed.get("strategy", "unknown"),
                    confidence=float(parsed.get("confidence", 0.0)),
                    reasoning=parsed.get("reasoning", ""),
                    suggested_operations=parsed.get("suggested_operations", []),
                    new_operations=parsed.get("new_operations", [])
                )
            else:
                # Fallback: treat as plain text
                return LLMResponse(
                    strategy="text_response",
                    confidence=0.5,
                    reasoning=response,
                    suggested_operations=[],
                    new_operations=[]
                )
                
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as plain text
            return LLMResponse(
                strategy="text_response",
                confidence=0.3,
                reasoning=response,
                suggested_operations=[],
                new_operations=[]
            )
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return LLMResponse(
                strategy="parse_error",
                confidence=0.0,
                reasoning=f"Failed to parse response: {e}",
                suggested_operations=[],
                new_operations=[]
            )
    
    def test_connection(self) -> bool:
        """Test if API server is available"""
        try:
            test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
            response = self.query_llm(test_prompt)
            return response != "LLM_ERROR" and len(response) > 0
        except Exception as e:
            logger.warning(f"API connection test failed: {e}")
            return False


def grid_object_to_description(obj) -> ObjectDescription:
    """Convert GridObject to ObjectDescription for LLM"""
    features = obj.features()
    
    return ObjectDescription(
        bbox=obj.bbox,
        area=features['area'],
        main_color=features['main_color'] or 0,
        color_histogram=obj.color_hist,
        shape_type=features['shape_type'],
        num_holes=features['num_holes'],
        touches_border=features['touches_border']
    )


def create_task_analysis(task_id: str, input_grid, output_grid, 
                        input_objects: List, output_objects: List) -> TaskAnalysis:
    """Create complete task analysis for LLM"""
    
    # Convert objects to descriptions
    input_descriptions = [grid_object_to_description(obj) for obj in input_objects]
    output_descriptions = [grid_object_to_description(obj) for obj in output_objects]
    
    # Create text representations of grids
    input_text = str(input_grid.pixels.tolist())
    output_text = str(output_grid.pixels.tolist())
    
    return TaskAnalysis(
        task_id=task_id,
        input_shape=input_grid.shape(),
        output_shape=output_grid.shape(),
        input_objects=input_descriptions,
        output_objects=output_descriptions,
        input_grid_text=input_text,
        output_grid_text=output_text
    )


# Global flag for LLM availability
LLM_AVAILABLE = True  # Will be set based on actual connection test 