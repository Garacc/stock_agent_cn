from datetime import datetime, timedelta
import argparse
from src.agents.valuation import valuation_agent
from src.agents.state import AgentState
from src.agents.sentiment import sentiment_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.market_data import market_data_agent
from src.agents.fundamentals import fundamentals_agent
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
import akshare as ak
import pandas as pd
from src.utils.logger_config import setup_logger, get_logger


##### Run the Hedge Fund #####
def run_hedge_fund(app, model: list, ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False, num_of_news: int = 5):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": num_of_news,
            },
            "metadata": {
                "model": model,
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content


def build_hedge_workflow():
    # Define the new workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("market_data_agent", market_data_agent)
    workflow.add_node("technical_analyst_agent", technical_analyst_agent)
    workflow.add_node("fundamentals_agent", fundamentals_agent)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    workflow.add_node("valuation_agent", valuation_agent)

    # Define the workflow
    workflow.set_entry_point("market_data_agent")
    workflow.add_edge("market_data_agent", "technical_analyst_agent")
    workflow.add_edge("market_data_agent", "fundamentals_agent")
    workflow.add_edge("market_data_agent", "sentiment_agent")
    workflow.add_edge("market_data_agent", "valuation_agent")
    workflow.add_edge("technical_analyst_agent", "risk_management_agent")
    workflow.add_edge("fundamentals_agent", "risk_management_agent")
    workflow.add_edge("sentiment_agent", "risk_management_agent")
    workflow.add_edge("valuation_agent", "risk_management_agent")
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    app = workflow.compile()
    
    return app

# Add this at the bottom of the file
if __name__ == "__main__":
    # Initialize logging system
    logger = get_logger()
    logger.info("启动StockAgent...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD). Defaults to 1 year before end date')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD). Defaults to yesterday')
    parser.add_argument('--show-reasoning', action='store_true',
                        help='Show reasoning from each agent')
    parser.add_argument('--num-of-news', type=int, default=5,
                        help='Number of news articles to analyze for sentiment (default: 5)')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial cash amount (default: 100,000)')
    parser.add_argument('--initial-position', type=int, default=0,
                        help='Initial stock position (default: 0)')
    parser.add_argument('--model', type=str, default='moonshot',
                        help='Model to use for chat completion (default: moonshot), use comma to separate multiple models.')

    args = parser.parse_args()

    logger.info("系统入参:")
    for arg_name, arg_value in vars(args).items():
        logger.info("{} = {}".format(arg_name, arg_value))

    # Set end date to yesterday if not specified
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday if not args.end_date else min(
        datetime.strptime(args.end_date, '%Y-%m-%d'), yesterday)

    # Set start date to one year before end date if not specified
    if not args.start_date:
        start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    # Validate dates
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")

    # Validate num_of_news
    if args.num_of_news < 1:
        raise ValueError("Number of news articles must be at least 1")
    if args.num_of_news > 100:
        raise ValueError("Number of news articles cannot exceed 100")
    
    logger.info("start_date: {}".format(start_date.strftime('%Y-%m-%d')))
    logger.info("end_date: {}".format(end_date.strftime('%Y-%m-%d')))

    # Configure portfolio
    portfolio = {
        "cash": args.initial_capital,
        "stock": args.initial_position
    }

    app = build_hedge_workflow()

    result = run_hedge_fund(
        app=app,
        model=args.model.split(','),
        ticker=args.ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        num_of_news=args.num_of_news
    )
    logger.info("Final Result:")
    logger.info(result)
