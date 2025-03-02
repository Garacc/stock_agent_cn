from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.utils.openrouter_config import get_chat_completion
from src.utils.logger_config import get_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
import json

from src.agents.state import AgentState, show_agent_reasoning

# 设置日志记录
logger = get_logger()

##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    logger.info("[PORTFOLIO_MANAGEMENT_AGENT] 开始执行投资组合管理Agent ...")
    model = state["metadata"]["model"]
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get the technical analyst, fundamentals agent, and risk management agent messages
    logger.info(f"{WAIT_ICON} 获取其他代理的分析结果...")
    try:
        technical_message = next(
            msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
        fundamentals_message = next(
            msg for msg in state["messages"] if msg.name == "fundamentals_agent")
        sentiment_message = next(
            msg for msg in state["messages"] if msg.name == "sentiment_agent")
        valuation_message = next(
            msg for msg in state["messages"] if msg.name == "valuation_agent")
        risk_message = next(
            msg for msg in state["messages"] if msg.name == "risk_management_agent")
        logger.info(f"{SUCCESS_ICON} 成功获取所有代理的分析结果")
    except Exception as e:
        logger.error(f"{ERROR_ICON} 获取代理分析结果失败: {e}")
        raise

    # Create the system message and user message
    logger.info(f"{WAIT_ICON} 准备系统消息和用户消息...")

    # Create the system message
    system_message = {
        "role": "system",
        "content": """You are a portfolio manager making final trading decisions.
            Your job is to make a trading decision based on the team's analysis while strictly adhering
            to risk management constraints.

            RISK MANAGEMENT CONSTRAINTS:
            - You MUST NOT exceed the max_position_size specified by the risk manager
            - You MUST follow the trading_action (buy/sell/hold) recommended by risk management
            - These are hard constraints that cannot be overridden by other signals

            When weighing the different signals for direction and timing:
            1. Valuation Analysis (35% weight)
               - Primary driver of fair value assessment
               - Determines if price offers good entry/exit point
            
            2. Fundamental Analysis (30% weight)
               - Business quality and growth assessment
               - Determines conviction in long-term potential
            
            3. Technical Analysis (25% weight)
               - Secondary confirmation
               - Helps with entry/exit timing
            
            4. Sentiment Analysis (10% weight)
               - Final consideration
               - Can influence sizing within risk limits
            
            The decision process should be:
            1. First check risk management constraints
            2. Then evaluate valuation signal
            3. Then evaluate fundamentals signal
            4. Use technical analysis for timing
            5. Consider sentiment for final adjustment
            
            Provide the following in your output:
            - "action": "buy" | "sell" | "hold",
            - "quantity": <positive integer>
            - "confidence": <float between 0 and 1>
            - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
            - "reasoning": <concise explanation of the decision including how you weighted the signals>

            Trading Rules:
            - Never exceed risk management position limits
            - Only buy if you have available cash
            - Only sell if you have shares to sell
            - Quantity must be ≤ current position for sells
            - Quantity must be ≤ max_position_size from risk management"""
    }

    # Create the user message
    user_message = {
        "role": "user",
        "content": f"""Based on the team's analysis below, make your trading decision.

            Technical Analysis Trading Signal: {technical_message.content}
            Fundamental Analysis Trading Signal: {fundamentals_message.content}
            Sentiment Analysis Trading Signal: {sentiment_message.content}
            Valuation Analysis Trading Signal: {valuation_message.content}
            Risk Management Trading Signal: {risk_message.content}

            Here is the current portfolio:
            Portfolio:
            Cash: {portfolio['cash']:.2f}
            Current Position: {portfolio['stock']} shares

            Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  Do not include any JSON markdown.

            Remember, the action must be either buy, sell, or hold.
            You can only buy if you have available cash.
            You can only sell if you have shares in the portfolio to sell."""
    }

    # Get the completion from OpenRouter
    logger.info(f"{WAIT_ICON} 正在获取LLM决策结果...")
    results = get_chat_completion([system_message, user_message], model)

    # 如果API调用失败，使用默认的保守决策
    if len(results) == 0:
        logger.warning(f"{ERROR_ICON} API调用失败，使用默认保守决策")
        default_result = json.dumps({
            "action": "hold",
            "quantity": 0,
            "confidence": 0.7,
            "agent_signals": [
                {
                    "agent_name": "technical_analysis",
                    "signal": "neutral",
                    "confidence": 0.0
                },
                {
                    "agent_name": "fundamental_analysis",
                    "signal": "bullish",
                    "confidence": 1.0
                },
                {
                    "agent_name": "sentiment_analysis",
                    "signal": "bullish",
                    "confidence": 0.6
                },
                {
                    "agent_name": "valuation_analysis",
                    "signal": "bearish",
                    "confidence": 0.67
                },
                {
                    "agent_name": "risk_management",
                    "signal": "hold",
                    "confidence": 1.0
                }
            ],
            "reasoning": "API error occurred. Following risk management signal to hold. This is a conservative decision based on the mixed signals: bullish fundamentals and sentiment vs bearish valuation, with neutral technicals."
        })
        # 创建一个默认结果字典，键为"default"
        results = {"default": default_result}

    # Create the portfolio management message
    messages = []
    for res_model, res_content in results.items():
        message = HumanMessage(
            content=res_content,
            name="portfolio_management_" + res_model,
        )

        # Show the decision if the flag is set
        if show_reasoning:
            show_agent_reasoning(message.content, f"Portfolio Management Agent with model {res_model}")
            
        try:
            decision = json.loads(res_content)  # 使用当前模型的结果
            logger.info(f"{SUCCESS_ICON} 模型 {res_model} 的投资决策: {decision['action']}, 数量: {decision['quantity']}, 置信度: {decision['confidence']}")
            messages.append(message)
        except Exception as e:
            logger.error(f"{ERROR_ICON} 解析模型 {res_model} 的决策结果失败: {e}")

    logger.info(f"{SUCCESS_ICON} [PORTFOLIO_MANAGEMENT_AGENT] 投资组合管理Agent执行完成")

    return {
        "messages": state["messages"] + messages,
        "data": state["data"],
    }


def format_decision(action: str, quantity: int, confidence: float, agent_signals: list, reasoning: str) -> dict:
    """Format the trading decision into a standardized output format.
    Think in English but output analysis in Chinese."""
    logger.info(f"{WAIT_ICON} 开始格式化交易决策...")

    try:
        # 获取各个agent的信号
        fundamental_signal = next(
            (signal for signal in agent_signals if signal["agent_name"] == "fundamental_analysis"), None)
        valuation_signal = next(
            (signal for signal in agent_signals if signal["agent_name"] == "valuation_analysis"), None)
        technical_signal = next(
            (signal for signal in agent_signals if signal["agent_name"] == "technical_analysis"), None)
        sentiment_signal = next(
            (signal for signal in agent_signals if signal["agent_name"] == "sentiment_analysis"), None)
        risk_signal = next(
            (signal for signal in agent_signals if signal["agent_name"] == "risk_management"), None)
        
        logger.info(f"{SUCCESS_ICON} 成功解析所有代理信号")

        # 转换信号为中文
        def signal_to_chinese(signal):
            if not signal:
                return "无数据"
            if signal["signal"] == "bullish":
                return "看多"
            elif signal["signal"] == "bearish":
                return "看空"
            return "中性"

        # 创建详细分析报告
        detailed_analysis = f"""
    ====================================
          投资分析报告
    ====================================

    一、策略分析

    1. 基本面分析 (权重30%):
       信号: {signal_to_chinese(fundamental_signal)}
       置信度: {fundamental_signal['confidence']*100:.0f}%
       要点: 
       - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据')}
       - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据')}
       - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据')}
       - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据')}

    2. 估值分析 (权重35%):
       信号: {signal_to_chinese(valuation_signal)}
       置信度: {valuation_signal['confidence']*100:.0f}%
       要点:
       - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据')}
       - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据')}

    3. 技术分析 (权重25%):
       信号: {signal_to_chinese(technical_signal)}
       置信度: {technical_signal['confidence']*100:.0f}%
       要点:
       - 趋势跟踪: ADX={technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', '无数据'):.2f}
       - 均值回归: RSI(14)={technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', '无数据'):.2f}
       - 动量指标: 
         * 1月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', '无数据'):.2%}
         * 3月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', '无数据'):.2%}
         * 6月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', '无数据'):.2%}
       - 波动性: {technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', '无数据'):.2%}

    4. 情绪分析 (权重10%):
       信号: {signal_to_chinese(sentiment_signal)}
       置信度: {sentiment_signal['confidence']*100:.0f}%
       分析: {sentiment_signal.get('reasoning', '无详细分析')}

    二、风险评估
    风险评分: {risk_signal.get('risk_score', '无数据')}/10
    主要指标:
    - 波动率: {risk_signal.get('risk_metrics', {}).get('volatility', '无数据')*100:.1f}%
    - 最大回撤: {risk_signal.get('risk_metrics', {}).get('max_drawdown', '无数据')*100:.1f}%
    - VaR(95%): {risk_signal.get('risk_metrics', {}).get('value_at_risk_95', '无数据')*100:.1f}%
    - 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据')}/10

    三、投资建议
    操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
    交易数量: {quantity}股
    决策置信度: {confidence*100:.0f}%

    四、决策依据
    {reasoning}

    ===================================="""

    except Exception as e:
        logger.error(f"{ERROR_ICON} 格式化决策结果失败: {e}")

    formatted_result = {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }
    
    logger.info(f"{SUCCESS_ICON} 决策格式化完成")
    return formatted_result  # 确保有返回语句
    
