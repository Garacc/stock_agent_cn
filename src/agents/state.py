from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage
from src.utils.logger_config import get_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

import json

# 设置日志记录
logger = get_logger()

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]



def show_agent_reasoning(output, agent_name):
    logger.info(f"{WAIT_ICON} {'=' * 10} {agent_name} 代理推理过程: {'=' * 10}")
    
    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation
    
    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        logger.info(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            logger.info(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            logger.info(output)
    
    logger.info(f"{SUCCESS_ICON} {'=' * 10} {agent_name} 代理推理完成 {'=' * 10}")