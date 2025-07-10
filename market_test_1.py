import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
import os
import google.generativeai as genai
from typing import Dict, List, Optional
import numpy as np

# Configure page
st.set_page_config(
    page_title="Money Management Dashboard",
    page_icon="💰",
    layout="wide"
)

# Configure Gemini AI
GOOGLE_API_KEY = "AIzaSyB46mW-7p4MIrKSe-oudQLpjxWli6XjVpE"
MODEL_NAME = "gemini-2.0-flash-001"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# File paths for data storage
HISTORY_FILE = "investment_history.json"
PROFILE_FILE = "user_profile.json"
TODAY_INVESTMENTS_FILE = "today_investments.json"

class MoneyManager:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Load historical data and user profile"""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except:
            self.history = []
        
        try:
            if os.path.exists(PROFILE_FILE):
                with open(PROFILE_FILE, 'r') as f:
                    self.profile = json.load(f)
            else:
                self.profile = {
                    "total_invested": 0,
                    "total_profit_loss": 0,
                    "win_rate": 0,
                    "avg_profit_percent": 0,
                    "risk_tolerance": "medium"
                }
        except:
            self.profile = {
                "total_invested": 0,
                "total_profit_loss": 0,
                "win_rate": 0,
                "avg_profit_percent": 0,
                "risk_tolerance": "medium"
            }
        
        # Load today's investments
        try:
            if os.path.exists(TODAY_INVESTMENTS_FILE):
                with open(TODAY_INVESTMENTS_FILE, 'r') as f:
                    data = json.load(f)
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    self.today_investments = data.get(today_str, [])
            else:
                self.today_investments = []
        except:
            self.today_investments = []
    
    def save_data(self):
        """Save data to files"""
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            with open(PROFILE_FILE, 'w') as f:
                json.dump(self.profile, f, indent=2)
            
            # Save today's investments
            today_str = datetime.now().strftime('%Y-%m-%d')
            try:
                if os.path.exists(TODAY_INVESTMENTS_FILE):
                    with open(TODAY_INVESTMENTS_FILE, 'r') as f:
                        all_today_data = json.load(f)
                else:
                    all_today_data = {}
                
                all_today_data[today_str] = self.today_investments
                
                with open(TODAY_INVESTMENTS_FILE, 'w') as f:
                    json.dump(all_today_data, f, indent=2)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error saving data: {e}")
    
    def add_today_investment(self, amount: float, investment_type: str, target_profit: float, notes: str = ""):
        """Add investment for today"""
        investment = {
            "time": datetime.now().strftime('%H:%M:%S'),
            "amount": amount,
            "investment_type": investment_type,
            "target_profit": target_profit,
            "notes": notes,
            "status": "active"
        }
        self.today_investments.append(investment)
        self.save_data()
    
    def get_today_total_invested(self):
        """Get total amount invested today"""
        return sum(inv["amount"] for inv in self.today_investments)
    
    def get_today_target_profit(self):
        """Get total target profit for today"""
        return sum(inv["amount"] * (inv["target_profit"] / 100) for inv in self.today_investments)
    
    def add_investment_record(self, amount: float, target_profit_percent: float, 
                            actual_profit_loss: float, actual_percent: float, 
                            investment_type: str, notes: str = ""):
        """Add new investment record"""
        record = {
            "date": datetime.now().isoformat(),
            "amount_invested": amount,
            "target_profit_percent": target_profit_percent,
            "actual_profit_loss": actual_profit_loss,
            "actual_percent": actual_percent,
            "investment_type": investment_type,
            "notes": notes,
            "final_amount": amount + actual_profit_loss
        }
        
        self.history.append(record)
        self.update_profile()
        self.save_data()
    
    def update_profile(self):
        """Update user profile based on history"""
        if not self.history:
            return
        
        total_invested = sum(record["amount_invested"] for record in self.history)
        total_profit_loss = sum(record["actual_profit_loss"] for record in self.history)
        
        wins = [r for r in self.history if r["actual_profit_loss"] > 0]
        win_rate = len(wins) / len(self.history) * 100 if self.history else 0
        
        avg_profit_percent = np.mean([r["actual_percent"] for r in self.history]) if self.history else 0
        
        self.profile.update({
            "total_invested": total_invested,
            "total_profit_loss": total_profit_loss,
            "win_rate": win_rate,
            "avg_profit_percent": avg_profit_percent,
            "total_trades": len(self.history)
        })
    
    def get_ai_recommendation(self, available_amount: float, target_profit: float, 
                            today_invested: float = 0, today_target: float = 0) -> pd.DataFrame:
        """Get AI-powered money management recommendation considering today's investments"""
        try:
            # Calculate current portfolio stats
            current_balance = available_amount
            win_rate = max(0.5, self.profile.get('win_rate', 60) / 100)  # Minimum 50% assumption
            avg_profit_percent = max(target_profit, self.profile.get('avg_profit_percent', target_profit))
            
            # Adjust recommendations based on today's investments
            risk_adjustment = 1.0
            if today_invested > 0:
                daily_risk_ratio = today_invested / (today_invested + available_amount)
                if daily_risk_ratio > 0.5:  # Already invested more than 50% of daily capital
                    risk_adjustment = 0.7  # Reduce risk
                elif daily_risk_ratio > 0.3:  # Moderate exposure
                    risk_adjustment = 0.85
            
            # Money management scenarios
            scenarios = []
            
            # Conservative approach (1-2% risk per trade)
            conservative_stake = current_balance * 0.015 * risk_adjustment
            conservative_trades = max(1, int(current_balance / conservative_stake)) if conservative_stake > 0 else 0
            scenarios.append(self._calculate_scenario("Conservative", current_balance, conservative_stake, 
                                                   min(conservative_trades, 10), win_rate, avg_profit_percent))
            
            # Moderate approach (3-5% risk per trade)
            moderate_stake = current_balance * 0.04 * risk_adjustment
            moderate_trades = max(1, int(current_balance / moderate_stake)) if moderate_stake > 0 else 0
            scenarios.append(self._calculate_scenario("Moderate", current_balance, moderate_stake, 
                                                   min(moderate_trades, 8), win_rate, avg_profit_percent))
            
            # Aggressive approach (7-10% risk per trade)
            aggressive_stake = current_balance * 0.08 * risk_adjustment
            aggressive_trades = max(1, int(current_balance / aggressive_stake)) if aggressive_stake > 0 else 0
            scenarios.append(self._calculate_scenario("Aggressive", current_balance, aggressive_stake, 
                                                   min(aggressive_trades, 6), win_rate, avg_profit_percent))
            
            # Target-based approach (calculate stake needed to reach remaining target)
            if today_target > 0:
                remaining_target = max(0, today_target - today_invested * (target_profit / 100))
                if remaining_target > 0 and target_profit > 0:
                    target_stake = remaining_target / (target_profit / 100)
                    target_trades = max(1, int(current_balance / target_stake)) if target_stake > 0 else 0
                    scenarios.append(self._calculate_scenario("Target-Based", current_balance, 
                                                           min(target_stake, current_balance), 
                                                           min(target_trades, 5), win_rate, target_profit))
            
            # Safe compound approach
            safe_stake = min(current_balance * 0.02, current_balance / 10)
            safe_trades = 10 if safe_stake > 0 else 0
            scenarios.append(self._calculate_scenario("Safe Compound", current_balance, safe_stake, 
                                                   safe_trades, win_rate, avg_profit_percent))
            
            return pd.DataFrame([s for s in scenarios if s["Trade_Stake"] > 0])
            
        except Exception as e:
            return pd.DataFrame([{
                "Strategy": "Error", 
                "Trade_Stake": 0, 
                "Total_Trades": 0,
                "Expected_Profit": 0,
                "Expected_Loss": 0,
                "Total_Capital_Growth": 0,
                "Growth_Percentage": 0,
                "Risk_Level": f"Error: {str(e)}"
            }])
    
    def _calculate_scenario(self, strategy_name: str, capital: float, stake: float, 
                          num_trades: int, win_rate: float, profit_percent: float) -> dict:
        """Calculate trading scenario results"""
        if stake <= 0 or num_trades <= 0 or capital <= 0:
            return {
                "Strategy": strategy_name,
                "Trade_Stake": 0,
                "Total_Trades": 0,
                "Expected_Profit": 0,
                "Expected_Loss": 0,
                "Total_Capital_Growth": 0,
                "Growth_Percentage": 0,
                "Risk_Level": "Invalid"
            }
        
        # Ensure stake doesn't exceed available capital
        stake = min(stake, capital)
        num_trades = min(num_trades, int(capital / stake))
        
        # Calculate expected outcomes
        winning_trades = num_trades * win_rate
        losing_trades = num_trades - winning_trades
        
        # For binary trading, typical payout is 70-90% of stake
        payout_ratio = 0.8  # 80% payout
        
        expected_profit = winning_trades * stake * payout_ratio
        expected_loss = losing_trades * stake
        
        net_result = expected_profit - expected_loss
        growth_percentage = (net_result / capital) * 100
        
        # Risk assessment
        risk_per_trade = (stake / capital) * 100
        if risk_per_trade <= 2:
            risk_level = "Low"
        elif risk_per_trade <= 5:
            risk_level = "Medium"
        elif risk_per_trade <= 10:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return {
            "Strategy": strategy_name,
            "Trade_Stake": round(stake, 2),
            "Total_Trades": int(num_trades),
            "Expected_Profit": round(expected_profit, 2),
            "Expected_Loss": round(expected_loss, 2),
            "Total_Capital_Growth": round(net_result, 2),
            "Growth_Percentage": round(growth_percentage, 2),
            "Risk_Level": risk_level
        }
    
    def get_remaining_day_strategy(self, remaining_amount: float, daily_target: float):
        """Get strategy for remaining amount to achieve daily target"""
        if remaining_amount <= 0 or daily_target <= 0:
            return "No strategy needed - insufficient funds or target already met"
        
        today_invested = self.get_today_total_invested()
        today_target_profit = self.get_today_target_profit()
        
        remaining_target = daily_target - today_target_profit
        
        if remaining_target <= 0:
            return "Daily target already achievable with current investments"
        
        # Calculate required profit percentage
        required_profit_percent = (remaining_target / remaining_amount) * 100
        
        context = f"""
        Current Situation:
        - Remaining amount to invest: ${remaining_amount:.2f}
        - Amount already invested today: ${today_invested:.2f}
        - Target profit remaining: ${remaining_target:.2f}
        - Required profit percentage: {required_profit_percent:.1f}%
        
        User's Historical Performance:
        - Win rate: {self.profile.get('win_rate', 0):.1f}%
        - Average profit: {self.profile.get('avg_profit_percent', 0):.1f}%
        
        Today's Investments:
        {json.dumps(self.today_investments, indent=2)}
        
        Please provide a practical strategy to achieve the remaining target with the available funds.
        Include specific recommendations for:
        1. Position sizing
        2. Number of trades
        3. Risk management
        4. Realistic expectations
        """
        
        try:
            response = model.generate_content(context)
            return response.text
        except Exception as e:
            return f"Error generating strategy: {e}"

# Initialize the money manager
if 'money_manager' not in st.session_state:
    st.session_state.money_manager = MoneyManager()

money_manager = st.session_state.money_manager

# Dashboard Header
st.title("💰 Money Management Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", 
                           ["Dashboard", "Today's Trading", "New Investment", "Add Results", "History", "AI Insights"])

if page == "Dashboard":
    st.header("📊 Investment Overview")
    
    # Today's summary
    today_invested = money_manager.get_today_total_invested()
    today_target = money_manager.get_today_target_profit()
    
    st.subheader("📅 Today's Trading Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Today's Invested", f"${today_invested:.2f}")
    
    with col2:
        st.metric("Today's Target Profit", f"${today_target:.2f}")
    
    with col3:
        st.metric("Active Trades", len(money_manager.today_investments))
    
    with col4:
        if today_invested > 0:
            target_percent = (today_target / today_invested) * 100
            st.metric("Target Return %", f"{target_percent:.1f}%")
        else:
            st.metric("Target Return %", "0%")
    
    # Overall portfolio metrics
    st.subheader("📈 Overall Portfolio")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invested", f"${money_manager.profile.get('total_invested', 0):,.2f}")
    
    with col2:
        profit_loss = money_manager.profile.get('total_profit_loss', 0)
        st.metric("Total P&L", f"${profit_loss:,.2f}", 
                 delta=f"{profit_loss:,.2f}")
    
    with col3:
        win_rate = money_manager.profile.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        avg_profit = money_manager.profile.get('avg_profit_percent', 0)
        st.metric("Avg Return %", f"{avg_profit:.2f}%")
    
    # Today's investments table
    if money_manager.today_investments:
        st.subheader("Today's Active Investments")
        today_df = pd.DataFrame(money_manager.today_investments)
        st.dataframe(today_df, use_container_width=True)
    
    # Charts
    if money_manager.history:
        st.subheader("Investment Performance")
        
        # Create DataFrame for plotting
        df = pd.DataFrame(money_manager.history)
        df['date'] = pd.to_datetime(df['date'])
        df['cumulative_pnl'] = df['actual_profit_loss'].cumsum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L over time
            fig_pnl = px.line(df, x='date', y='cumulative_pnl', 
                             title='Cumulative P&L Over Time')
            fig_pnl.update_traces(line_color='green' if df['cumulative_pnl'].iloc[-1] > 0 else 'red')
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        with col2:
            # Investment distribution
            fig_pie = px.pie(df, values='amount_invested', names='investment_type',
                            title='Investment Distribution by Type')
            st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Today's Trading":
    st.header("🎯 Today's Trading Manager")
    
    # Add new investment for today
    st.subheader("Add Today's Investment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Investment Amount ($)", min_value=0.01, value=50.0, step=0.01)
        investment_type = st.selectbox("Investment Type", 
                                     ["Binary Trading", "Stocks", "Crypto", "Forex", "Options", "Futures", "Other"])
        target_profit = st.number_input("Target Profit (%)", min_value=0.01, value=5.0, step=0.01)
        notes = st.text_input("Notes (optional)")
        
        if st.button("Add Investment", type="primary"):
            money_manager.add_today_investment(amount, investment_type, target_profit, notes)
            st.success(f"Added ${amount:.2f} investment with {target_profit:.1f}% target!")
            st.rerun()
    
    with col2:
        st.subheader("Today's Summary")
        today_invested = money_manager.get_today_total_invested()
        today_target = money_manager.get_today_target_profit()
        
        if today_invested > 0:
            st.metric("Total Invested Today", f"${today_invested:.2f}")
            st.metric("Target Profit Today", f"${today_target:.2f}")
            st.metric("Number of Trades", len(money_manager.today_investments))
            
            target_percent = (today_target / today_invested) * 100
            st.metric("Overall Target %", f"{target_percent:.1f}%")
    
    # Money allocation for remaining funds
    st.subheader("💡 Remaining Money Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remaining_amount = st.number_input("Remaining Amount to Invest ($)", min_value=0.0, value=100.0, step=0.01)
        daily_target = st.number_input("Daily Target Profit ($)", min_value=0.0, value=50.0, step=0.01)
        
        if st.button("Get Strategy for Remaining Money", type="primary"):
            if remaining_amount > 0:
                with st.spinner("Calculating optimal strategy..."):
                    strategy = money_manager.get_remaining_day_strategy(remaining_amount, daily_target)
                    st.session_state.remaining_strategy = strategy
                    
                    # Also get recommendation table
                    recommendation_df = money_manager.get_ai_recommendation(
                        remaining_amount, 
                        (daily_target / remaining_amount) * 100 if remaining_amount > 0 else 5,
                        money_manager.get_today_total_invested(),
                        money_manager.get_today_target_profit()
                    )
                    st.session_state.remaining_recommendation = recommendation_df
    
    with col2:
        if hasattr(st.session_state, 'remaining_strategy'):
            st.subheader("🤖 AI Strategy")
            st.markdown(st.session_state.remaining_strategy)
    
    # Display recommendation table
    if hasattr(st.session_state, 'remaining_recommendation') and isinstance(st.session_state.remaining_recommendation, pd.DataFrame):
        st.subheader("💼 Money Management Options")
        st.dataframe(st.session_state.remaining_recommendation, use_container_width=True)
        
        # Best strategy highlight
        if not st.session_state.remaining_recommendation.empty:
            best_strategy = st.session_state.remaining_recommendation.loc[
                st.session_state.remaining_recommendation['Growth_Percentage'].idxmax()]
            
            st.success(f"🏆 **Recommended Strategy**: {best_strategy['Strategy']} - "
                      f"Stake ${best_strategy['Trade_Stake']:.2f} per trade, "
                      f"Expected {best_strategy['Growth_Percentage']:.1f}% growth")
    
    # Display today's investments
    if money_manager.today_investments:
        st.subheader("📋 Today's Investments")
        today_df = pd.DataFrame(money_manager.today_investments)
        
        # Add calculated columns
        today_df['Target_Profit_$'] = today_df['amount'] * (today_df['target_profit'] / 100)
        today_df['Expected_Return'] = today_df['amount'] + today_df['Target_Profit_$']
        
        st.dataframe(today_df, use_container_width=True)
        
        # Today's analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_amount = today_df['amount'].sum()
            st.metric("Total Amount", f"${total_amount:.2f}")
        
        with col2:
            total_target = today_df['Target_Profit_$'].sum()
            st.metric("Total Target Profit", f"${total_target:.2f}")
        
        with col3:
            if total_amount > 0:
                avg_target_percent = (total_target / total_amount) * 100
                st.metric("Average Target %", f"{avg_target_percent:.1f}%")

elif page == "New Investment":
    st.header("🎯 Plan New Investment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Investment Details")
        amount = st.number_input("Amount to Invest ($)", min_value=0.01, value=100.0, step=0.01)
        target_profit = st.number_input("Target Profit (%)", min_value=0.01, value=5.0, step=0.01)
        investment_type = st.selectbox("Investment Type", 
                                     ["Binary Trading", "Stocks", "Crypto", "Forex", "Options", "Futures", "Other"])
        
        if st.button("Get AI Recommendation", type="primary"):
            with st.spinner("Calculating money management strategies..."):
                recommendation_df = money_manager.get_ai_recommendation(amount, target_profit)
                st.session_state.ai_recommendation = recommendation_df
    
    with col2:
        st.subheader("Quick Stats")
        if money_manager.history:
            similar_investments = [r for r in money_manager.history 
                                 if abs(r['amount_invested'] - amount) <= amount * 0.2]
            if similar_investments:
                avg_return = np.mean([r['actual_percent'] for r in similar_investments])
                st.info(f"Similar investments averaged {avg_return:.2f}% return")
        
        # Risk calculator
        risk_amount = amount * (target_profit / 100)
        st.warning(f"Potential gain: ${risk_amount:.2f}")
        
        # Position sizing suggestion
        if money_manager.profile.get('total_invested', 0) > 0:
            portfolio_percent = (amount / money_manager.profile['total_invested']) * 100
            st.info(f"This represents {portfolio_percent:.1f}% of your historical portfolio")
    
    # Display AI recommendation
    if hasattr(st.session_state, 'ai_recommendation') and isinstance(st.session_state.ai_recommendation, pd.DataFrame):
        st.subheader("🤖 Money Management Strategies")
        
        # Display the recommendation table
        st.dataframe(st.session_state.ai_recommendation, use_container_width=True)
        
        # Additional insights
        st.subheader("📊 Strategy Analysis")
        
        # Find best strategy
        if not st.session_state.ai_recommendation.empty:
            best_strategy = st.session_state.ai_recommendation.loc[
                st.session_state.ai_recommendation['Growth_Percentage'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Strategy", best_strategy['Strategy'])
                st.metric("Expected Growth", f"{best_strategy['Growth_Percentage']:.1f}%")
            
            with col2:
                st.metric("Recommended Stake", f"${best_strategy['Trade_Stake']:.2f}")
                st.metric("Total Trades", f"{best_strategy['Total_Trades']}")
            
            with col3:
                st.metric("Expected Profit", f"${best_strategy['Expected_Profit']:.2f}")
                st.metric("Risk Level", best_strategy['Risk_Level'])
            
            # Strategy comparison chart
            fig = px.bar(st.session_state.ai_recommendation, 
                        x='Strategy', y='Growth_Percentage',
                        title='Expected Growth % by Strategy',
                        color='Risk_Level',
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Very High': 'darkred'})
            st.plotly_chart(fig, use_container_width=True)

elif page == "Add Results":
    st.header("📈 Record Investment Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Investment Results")
        
        # Option to add results for today's investments or new completed investment
        option = st.radio("Choose option:", 
                         ["Record result for today's investment", 
                          "Add new completed investment"])
        
        if option == "Record result for today's investment" and money_manager.today_investments:
            # Show today's investments for selection
            today_options = []
            for i, inv in enumerate(money_manager.today_investments):
                today_options.append(f"${inv['amount']:.2f} - {inv['investment_type']} ({inv['target_profit']:.1f}%)")
            
            selected_inv = st.selectbox("Select Investment:", today_options)
            selected_idx = today_options.index(selected_inv)
            
            selected_investment = money_manager.today_investments[selected_idx]
            amount = selected_investment['amount']
            target_profit = selected_investment['target_profit']
            investment_type = selected_investment['investment_type']
            
            st.info(f"Recording result for: ${amount:.2f} {investment_type} trade")
            
        else:
            # Add new completed investment
            amount = st.number_input("Amount Invested ($)", min_value=0.01, value=100.0)
            target_profit = st.number_input("Target Profit (%) you had", min_value=-100.0, value=5.0)
            investment_type = st.selectbox("Investment Type", 
                                         ["Binary Trading", "Stocks", "Crypto", "Forex", "Options", "Futures", "Other"])
        
        actual_profit_loss = st.number_input("Actual Profit/Loss ($)", value=0.0)
        actual_percent = st.number_input("Actual Profit/Loss (%)", value=0.0)
        notes = st.text_area("Notes (optional)")
        
        if st.button("Add Investment Record", type="primary"):
            money_manager.add_investment_record(
                amount, target_profit, actual_profit_loss, 
                actual_percent, investment_type, notes
            )
            st.success("Investment record added successfully!")
            st.rerun()
    
    with col2:
        st.subheader("Performance Analysis")
        
        if actual_profit_loss != 0:
            if actual_profit_loss > 0:
                st.success(f"Profit: ${actual_profit_loss:.2f} ({actual_percent:.2f}%)")
            else:
                st.error(f"Loss: ${actual_profit_loss:.2f} ({actual_percent:.2f}%)")
            
            # Performance vs target
            if 'amount' in locals() and 'target_profit' in locals():
                target_amount = amount * (target_profit / 100)
                vs_target = actual_profit_loss - target_amount
                if vs_target > 0:
                    st.info(f"Beat target by ${vs_target:.2f}")
                else:
                    st.warning(f"Missed target by ${abs(vs_target):.2f}")

elif page == "History":
    st.header("📋 Investment History")
    
    if not money_manager.history:
        st.info("No investment history found.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            investment_types = list(set([r['investment_type'] for r in money_manager.history]))
            selected_types = st.multiselect("Filter by Type", investment_types, default=investment_types)
        
        with col2:
            date_range = st.date_input("Date Range", value=(date.today().replace(day=1), date.today()))
        
        with col3:
            show_only = st.selectbox("Show Only", ["All", "Profits", "Losses"])
        
        # Filter data
        filtered_history = money_manager.history
        
        if selected_types:
            filtered_history = [r for r in filtered_history if r['investment_type'] in selected_types]
        
        if show_only == "Profits":
            filtered_history = [r for r in filtered_history if r['actual_profit_loss'] > 0]
        elif show_only == "Losses":
            filtered_history = [r for r in filtered_history if r['actual_profit_loss'] < 0]
        
        # Display filtered history
        if filtered_history:
            df = pd.DataFrame(filtered_history)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(df[['date', 'amount_invested', 'target_profit_percent', 
                           'actual_profit_loss', 'actual_percent', 'investment_type', 'notes']], 
                        use_container_width=True)
            
            # Summary stats for filtered data
            st.subheader("Filtered Results Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_invested = sum(r['amount_invested'] for r in filtered_history)
                st.metric("Total Invested", f"${total_invested:,.2f}")
            
            with col2:
                total_pnl = sum(r['actual_profit_loss'] for r in filtered_history)
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            
            with col3:
                win_rate = len([r for r in filtered_history if r['actual_profit_loss'] > 0]) / len(filtered_history) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.info("No records match the selected filters.")

elif page == "AI Insights":
    st.header("🤖 AI-Powered Insights")
    
    if not money_manager.history:
        st.info("No investment history available for analysis.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Analysis")
            
            if st.button("Generate Portfolio Analysis", type="primary"):
                with st.spinner("Analyzing your portfolio..."):
                    context = f"""
                    Analyze this investment portfolio and provide insights:
                    
                    Portfolio Summary:
                    - Total Investments: {len(money_manager.history)}
                    - Total Amount Invested: ${money_manager.profile.get('total_invested', 0):,.2f}
                    - Total P&L: ${money_manager.profile.get('total_profit_loss', 0):,.2f}
                    - Win Rate: {money_manager.profile.get('win_rate', 0):.1f}%
                    - Average Return: {money_manager.profile.get('avg_profit_percent', 0):.2f}%
                    
                    Today's Trading Activity:
                    - Today's Investments: {len(money_manager.today_investments)}
                    - Today's Total Invested: ${money_manager.get_today_total_invested():.2f}
                    - Today's Target Profit: ${money_manager.get_today_target_profit():.2f}
                    
                    Recent Investment History:
                    {json.dumps(money_manager.history[-10:], indent=2)}
                    
                    Please provide:
                    1. Portfolio performance analysis
                    2. Risk assessment
                    3. Diversification recommendations
                    4. Areas for improvement
                    5. Future investment strategy suggestions
                    6. Analysis of today's trading approach
                    """
                    
                    try:
                        response = model.generate_content(context)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error generating analysis: {e}")
        
        with col2:
            st.subheader("Ask AI Anything")
            
            question = st.text_area("Ask about your investments:")
            
            if st.button("Get AI Answer") and question:
                with st.spinner("Getting AI response..."):
                    context = f"""
                    User Question: {question}
                    
                    User's Investment Data:
                    {json.dumps(money_manager.profile, indent=2)}
                    
                    Today's Investments:
                    {json.dumps(money_manager.today_investments, indent=2)}
                    
                    Recent History:
                    {json.dumps(money_manager.history[-5:], indent=2)}
                    
                    Please provide a helpful, specific answer based on their investment data.
                    """
                    
                    try:
                        response = model.generate_content(context)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error: {e}")

# Quick Actions Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")

# Quick add investment
with st.sidebar.expander("Quick Add Investment"):
    quick_amount = st.number_input("Amount ($)", min_value=0.01, value=50.0, step=0.01, key="quick_amount")
    quick_type = st.selectbox("Type", ["Binary Trading", "Crypto", "Forex", "Stocks"], key="quick_type")
    quick_target = st.number_input("Target %", min_value=0.01, value=5.0, step=0.01, key="quick_target")
    
    if st.button("Add to Today", key="quick_add"):
        money_manager.add_today_investment(quick_amount, quick_type, quick_target, "Quick add")
        st.success("Added!")
        st.rerun()

# Today's summary in sidebar
if money_manager.today_investments:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Today's Summary")
    st.sidebar.metric("Invested", f"${money_manager.get_today_total_invested():.2f}")
    st.sidebar.metric("Target Profit", f"${money_manager.get_today_target_profit():.2f}")
    st.sidebar.metric("Trades", len(money_manager.today_investments))

# Footer
st.markdown("---")
st.markdown("💡 **New Features**: Track today's investments and get AI strategies for your remaining money!")
st.markdown("📊 **Today's Trading**: Add investments as you make them and get real-time recommendations")
st.markdown("⚠️ **Disclaimer**: This is for educational purposes only. Always consult with a financial advisor before making investment decisions.")