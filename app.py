import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import base64
import io

# Set page config with Mario theme
st.set_page_config(
    page_title="Super Budget Bros",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'budget_limits' not in st.session_state:
    st.session_state.budget_limits = {}
if 'savings_goal' not in st.session_state:
    st.session_state.savings_goal = 0
if 'game_score' not in st.session_state:
    st.session_state.game_score = 0
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = 'Mario'
if 'selected_track' not in st.session_state:
    st.session_state.selected_track = 'Mushroom Kingdom'
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'risk_tolerance': 'Moderate',
        'savings_goal': 20,
        'investment_preference': []
    }

# Mario Kart theme configuration
THEME = {
    'light': {
        'background': '#F8F8F8',
        'text': '#000000',
        'primary': '#E52521',  # Mario Red
        'secondary': '#F8D210',  # Mario Yellow
        'accent': '#0066CC',  # Mario Blue
        'success': '#4CAF50',  # Luigi Green
        'warning': '#FF9800',  # Orange
        'danger': '#F44336',  # Red
        'panel': '#FFE4B5',  # Retro panel color
        'border': '#8B4513'  # Retro border color
    },
    'dark': {
        'background': '#1E1E1E',
        'text': '#FFFFFF',
        'primary': '#E52521',
        'secondary': '#F8D210',
        'accent': '#0066CC',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'danger': '#F44336',
        'panel': '#2D2D2D',
        'border': '#4A4A4A'
    }
}

# Character and Track configuration
CHARACTERS = {
    'Mario': {
        'color': '#E52521',
        'icon': '👨‍🔧',
        'description': 'The classic racer! Balanced stats for steady financial growth.',
        'tracks': ['Mushroom Kingdom', 'Peach\'s Castle', 'Bowser\'s Castle'],
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Mario/3D/mario_3d.png',
        'fallback': '👨‍🔧'
    },
    'Luigi': {
        'color': '#4CAF50',
        'icon': '👨‍🌾',
        'description': 'The careful saver! Great at managing expenses and building wealth.',
        'tracks': ['Rainbow Road', 'Luigi\'s Mansion', 'Ghost Valley'],
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Luigi/3D/luigi_3d.png',
        'fallback': '👨‍🌾'
    },
    'Peach': {
        'color': '#FF69B4',
        'icon': '👸',
        'description': 'The strategic investor! Excels at making smart financial decisions.',
        'tracks': ['Peach\'s Garden', 'Royal Raceway', 'Cake Circuit'],
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Peach/3D/peach_3d.png',
        'fallback': '👸'
    },
    'Yoshi': {
        'color': '#4CAF50',
        'icon': '🦖',
        'description': 'The consistent performer! Great at maintaining steady financial habits.',
        'tracks': ['Yoshi Valley', 'Dino Dino Jungle', 'Egg Circuit'],
        'image': 'https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Yoshi/3D/yoshi_3d.png',
        'fallback': '🦖'
    }
}

# Apply theme with character-specific styling
def apply_theme():
    theme = THEME['dark' if st.session_state.dark_mode else 'light']
    character = CHARACTERS[st.session_state.selected_character]
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        
        .stApp {{
            background-color: {theme['background']};
            color: {theme['text']};
            font-family: 'Press Start 2P', cursive;
            background-image: url('https://mario.wiki.gallery/images/thumb/3/3e/MK8_Mario_Icon.png/120px-MK8_Mario_Icon.png');
            background-repeat: no-repeat;
            background-position: right bottom;
            background-size: 200px;
            background-attachment: fixed;
        }}
        
        /* Main content area */
        .main .block-container {{
            background-color: {theme['panel']};
            border: 4px solid {theme['border']};
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 5px 5px 0px {theme['border']};
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background-color: {theme['panel']} !important;
            border-right: 4px solid {theme['border']};
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: {character['color']};
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            border: 3px solid {theme['border']};
            font-family: 'Press Start 2P', cursive;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 3px 3px 0px {theme['border']};
            transition: all 0.2s;
        }}
        
        .stButton>button:hover {{
            transform: translate(2px, 2px);
            box-shadow: 1px 1px 0px {theme['border']};
        }}
        
        /* Input fields */
        .stSelectbox, .stNumberInput, .stTextInput {{
            background-color: {theme['panel']};
            color: {theme['text']};
            border: 2px solid {theme['border']};
            border-radius: 10px;
            font-family: 'Press Start 2P', cursive;
            box-shadow: 3px 3px 0px {theme['border']};
        }}
        
        /* Headers */
        .stMarkdown h1 {{
            color: {character['color']};
            font-family: 'Press Start 2P', cursive;
            text-align: center;
            text-shadow: 3px 3px 0px {theme['border']};
            font-size: 2.5rem;
            margin-bottom: 2rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        
        .stMarkdown h2 {{
            color: {character['color']};
            font-family: 'Press Start 2P', cursive;
            text-shadow: 2px 2px 0px {theme['border']};
            font-size: 1.8rem;
            margin-top: 2rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Metrics */
        .stMetric {{
            background-color: {theme['panel']};
            border: 3px solid {theme['border']};
            border-radius: 15px;
            padding: 15px;
            box-shadow: 3px 3px 0px {theme['border']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {theme['panel']};
            border: 2px solid {theme['border']};
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-family: 'Press Start 2P', cursive;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {character['color']} !important;
            color: white !important;
        }}
        
        /* Track selector */
        .track-selector {{
            background-color: {character['color']}20;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            border: 3px solid {theme['border']};
            box-shadow: 3px 3px 0px {theme['border']};
        }}
        
        /* Progress bars */
        .stProgress > div > div > div {{
            background-color: {character['color']};
            border-radius: 10px;
        }}
        
        /* Tables */
        .stDataFrame {{
            background-color: {theme['panel']};
            border: 3px solid {theme['border']};
            border-radius: 15px;
            padding: 1rem;
        }}
        
        /* Radio buttons */
        .stRadio > div {{
            background-color: {theme['panel']};
            border: 2px solid {theme['border']};
            border-radius: 10px;
            padding: 1rem;
        }}
        
        /* Checkboxes */
        .stCheckbox > div {{
            background-color: {theme['panel']};
            border: 2px solid {theme['border']};
            border-radius: 10px;
            padding: 1rem;
        }}
        </style>
    """, unsafe_allow_html=True)

# Add this function after the apply_theme function
def style_visualization(fig, character):
    """Apply Mario Kart themed styling to Plotly figures"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='Press Start 2P, cursive',
            size=12,
            color=character['color']
        ),
        title=dict(
            font=dict(
                family='Press Start 2P, cursive',
                size=16,
                color=character['color']
            )
        ),
        legend=dict(
            font=dict(
                family='Press Start 2P, cursive',
                size=10
            )
        )
    )
    return fig

# ML Functions
def analyze_budget_with_ml(df):
    # Prepare data for ML
    X = pd.get_dummies(df[['Category', 'Type']])
    y = df['Amount']
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_
    })
    
    # Predict future expenses
    future_dates = pd.date_range(start=df['Date'].max(), periods=6, freq='ME')
    future_predictions = []
    
    for date in future_dates:
        features = pd.DataFrame({
            'Category': df['Category'].unique(),
            'Type': ['Expense'] * len(df['Category'].unique())
        })
        features_encoded = pd.get_dummies(features)
        
        for col in X.columns:
            if col not in features_encoded.columns:
                features_encoded[col] = 0
        
        features_encoded = features_encoded[X.columns]
        predictions = model.predict(features_encoded)
        future_predictions.append(predictions)
    
    return model, feature_importance, future_dates, future_predictions

def detect_anomalies(df):
    # Prepare data for anomaly detection
    expense_data = df[df['Type'] == 'Expense'].groupby(['Date', 'Category'])['Amount'].sum().reset_index()
    
    # Calculate rolling statistics for each category
    anomalies = []
    for category in expense_data['Category'].unique():
        category_data = expense_data[expense_data['Category'] == category]
        
        # Calculate rolling mean and standard deviation
        rolling_mean = category_data['Amount'].rolling(window=3, min_periods=1).mean()
        rolling_std = category_data['Amount'].rolling(window=3, min_periods=1).std()
        
        # Identify anomalies (3 standard deviations from mean)
        upper_bound = rolling_mean + (3 * rolling_std)
        lower_bound = rolling_mean - (3 * rolling_std)
        
        # Find anomalies
        category_anomalies = category_data[
            (category_data['Amount'] > upper_bound) | 
            (category_data['Amount'] < lower_bound)
        ]
        
        if not category_anomalies.empty:
            for _, row in category_anomalies.iterrows():
                anomalies.append({
                    'Date': row['Date'],
                    'Category': category,
                    'Amount': row['Amount'],
                    'Expected Range': f"${lower_bound.iloc[-1]:,.2f} - ${upper_bound.iloc[-1]:,.2f}",
                    'Deviation': f"{(row['Amount'] - rolling_mean.iloc[-1]) / rolling_mean.iloc[-1] * 100:.1f}%"
                })
    
    return pd.DataFrame(anomalies)

def check_loan_eligibility(income, expenses, credit_score, loan_amount):
    # Prepare features
    X = np.array([[income, expenses, credit_score, loan_amount]])
    
    # Train logistic regression model
    model = LogisticRegression()
    # Mock training data
    X_train = np.array([
        [50000, 30000, 700, 10000],
        [75000, 40000, 800, 20000],
        [100000, 50000, 900, 30000],
        [30000, 25000, 600, 5000],
        [40000, 35000, 650, 8000]
    ])
    y_train = np.array([1, 1, 1, 0, 0])
    model.fit(X_train, y_train)
    
    # Predict eligibility
    probability = model.predict_proba(X)[0][1]
    return probability > 0.5, probability

# Sample data generation
def generate_sample_data():
    start_date = datetime.now() - timedelta(days=180)
    dates = pd.date_range(start=start_date, periods=180, freq='D')
    
    transactions = []
    categories = {
        'Income': ['Salary', 'Freelance', 'Investments', 'Other Income'],
        'Expense': ['Groceries', 'Utilities', 'Entertainment', 'Transportation', 'Healthcare', 'Education', 'Shopping']
    }
    
    base_salary = 75000
    expense_patterns = {
        'Groceries': {'base': 2000, 'variation': 500, 'frequency': 0.3},
        'Utilities': {'base': 1500, 'variation': 300, 'frequency': 0.1},
        'Entertainment': {'base': 1000, 'variation': 400, 'frequency': 0.2},
        'Transportation': {'base': 1200, 'variation': 300, 'frequency': 0.2},
        'Healthcare': {'base': 800, 'variation': 200, 'frequency': 0.1},
        'Education': {'base': 1500, 'variation': 500, 'frequency': 0.1},
        'Shopping': {'base': 2000, 'variation': 800, 'frequency': 0.2}
    }
    
    for date in dates:
        if date.day == 1:
            salary = base_salary * (1 + np.random.normal(0, 0.05))
            transactions.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Category': 'Salary',
                'Description': 'Monthly Salary',
                'Amount': salary,
                'Type': 'Income'
            })
        
        for category, pattern in expense_patterns.items():
            if np.random.random() < pattern['frequency']:
                amount = pattern['base'] * (1 + np.random.normal(0, 0.1))
                amount += np.random.normal(0, pattern['variation'])
                transactions.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Category': category,
                    'Description': f'{category} expense',
                    'Amount': abs(amount),
                    'Type': 'Expense'
                })
    
    return pd.DataFrame(transactions)

# Add new function for risk assessment
def assess_risk_tolerance(answers):
    # Convert answers to numeric scores
    risk_questions = {
        "How do you feel about investment risk?": {
            "Very uncomfortable": 1,
            "Somewhat uncomfortable": 2,
            "Neutral": 3,
            "Comfortable": 4,
            "Very comfortable": 5
        },
        "How would you react to a 20% market drop?": {
            "Sell everything": 1,
            "Sell some": 2,
            "Hold": 3,
            "Buy more": 4,
            "Buy aggressively": 5
        },
        "Your investment time horizon is:": {
            "Less than 1 year": 1,
            "1-3 years": 2,
            "3-5 years": 3,
            "5-10 years": 4,
            "More than 10 years": 5
        }
    }
    
    score = 0
    for question, answer in answers.items():
        score += risk_questions[question][answer]
    
    if score <= 5:
        return "Conservative", "You prefer low-risk investments and stable returns."
    elif score <= 10:
        return "Moderate", "You're willing to take some risks for better returns."
    else:
        return "Aggressive", "You're comfortable with high-risk, high-reward investments."

# Add new function for budget recommendations
def generate_recommendations(df, preferences):
    recommendations = []
    
    # Calculate current savings rate
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    current_savings_rate = (total_income - total_expenses) / total_income * 100
    
    # Savings goal analysis
    if preferences['savings_goal']:
        target_savings = preferences['savings_goal']
        if current_savings_rate < target_savings:
            recommendations.append(f"⚠️ Your current savings rate ({current_savings_rate:.1f}%) is below your target ({target_savings}%). Consider reducing expenses in high-spending categories.")
    
    # Risk-based recommendations
    if preferences['risk_tolerance'] == "Conservative":
        recommendations.append("💡 Consider high-yield savings accounts or CDs for your savings.")
    elif preferences['risk_tolerance'] == "Moderate":
        recommendations.append("💡 Consider a balanced portfolio of stocks and bonds.")
    else:
        recommendations.append("💡 Consider growth stocks and index funds for higher returns.")
    
    # Category-specific recommendations
    expense_by_category = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
    avg_expense = expense_by_category.mean()
    
    for category, amount in expense_by_category.items():
        if amount > avg_expense * 1.2:  # 20% above average
            recommendations.append(f"📊 Consider reducing {category} expenses (currently ${amount:,.2f})")
    
    return recommendations

def analyze_budget_breakdown(df):
    try:
        # Calculate monthly averages
        monthly_income = df[df['Type'] == 'Income'].groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum().mean()
        monthly_expenses = df[df['Type'] == 'Expense'].groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum().mean()
        
        # Calculate category-wise spending
        expense_by_category = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
        total_expenses = expense_by_category.sum()
        
        # Calculate percentages and recommendations
        breakdown = []
        for category, amount in expense_by_category.items():
            percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            breakdown.append({
                'Category': category,
                'Amount': amount,
                'Percentage': percentage,
                'Monthly Average': amount / 6,  # Assuming 6 months of data
                'Recommendation': get_category_recommendation(category, percentage, monthly_income)
            })
        
        return pd.DataFrame(breakdown)
    except Exception as e:
        st.error(f"Error in budget analysis: {str(e)}")
        return pd.DataFrame()

def get_category_recommendation(category, percentage, monthly_income):
    try:
        # Standard budget percentages based on financial planning guidelines
        standard_percentages = {
            'Housing': 30,
            'Transportation': 15,
            'Food': 15,
            'Utilities': 5,
            'Healthcare': 5,
            'Insurance': 5,
            'Debt Payments': 10,
            'Entertainment': 5,
            'Personal Care': 5,
            'Miscellaneous': 5
        }
        
        # Map our categories to standard categories
        category_mapping = {
            'Groceries': 'Food',
            'Utilities': 'Utilities',
            'Entertainment': 'Entertainment',
            'Transportation': 'Transportation',
            'Healthcare': 'Healthcare',
            'Education': 'Miscellaneous',
            'Shopping': 'Miscellaneous',
            'Rent': 'Housing',
            'Mortgage': 'Housing',
            'Insurance': 'Insurance',
            'Loan Payment': 'Debt Payments'
        }
        
        mapped_category = category_mapping.get(category, 'Miscellaneous')
        standard_percentage = standard_percentages.get(mapped_category, 5)
        
        if percentage > standard_percentage * 1.2:  # 20% above standard
            return f"⚠️ Above standard ({standard_percentage}%). Consider reducing spending in this category."
        elif percentage < standard_percentage * 0.8:  # 20% below standard
            return f"✅ Below standard ({standard_percentage}%). Good job managing this category."
        else:
            return f"✅ Within standard range ({standard_percentage}%)."
    except Exception as e:
        return f"Error in recommendation: {str(e)}"

def generate_report(df, report_type, preferences=None):
    try:
        if report_type == "Summary Report":
            # Calculate key metrics
            total_income = df[df['Type'] == 'Income']['Amount'].sum()
            total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
            net_savings = total_income - total_expenses
            savings_rate = (net_savings / total_income) * 100 if total_income > 0 else 0
            
            # Monthly trends
            monthly_data = df.groupby(pd.Grouper(key='Date', freq='ME')).agg({
                'Amount': lambda x: x[df['Type'] == 'Income'].sum() - x[df['Type'] == 'Expense'].sum()
            }).reset_index()
            
            # Create report DataFrame
            report_data = {
                'Metric': [
                    'Total Income',
                    'Total Expenses',
                    'Net Savings',
                    'Savings Rate',
                    'Average Monthly Income',
                    'Average Monthly Expenses',
                    'Average Monthly Savings'
                ],
                'Amount': [
                    total_income,
                    total_expenses,
                    net_savings,
                    savings_rate,
                    monthly_data['Amount'].mean(),
                    df[df['Type'] == 'Expense'].groupby(pd.Grouper(key='Date', freq='ME'))['Amount'].sum().mean(),
                    monthly_data['Amount'].mean()
                ]
            }
            return pd.DataFrame(report_data)
        
        elif report_type == "Detailed Transactions":
            # Sort by date and add month column
            df_sorted = df.sort_values('Date').copy()
            df_sorted['Month'] = df_sorted['Date'].dt.strftime('%Y-%m')
            # Ensure all columns are properly formatted
            df_sorted['Amount'] = df_sorted['Amount'].round(2)
            return df_sorted
        
        else:  # Budget Analysis
            budget_breakdown = analyze_budget_breakdown(df)
            if preferences and 'budget_limits' in preferences:
                budget_breakdown['Target Percentage'] = [
                    preferences['budget_limits'].get(cat, 0) for cat in budget_breakdown['Category']
                ]
            # Ensure all numeric columns are properly formatted
            budget_breakdown['Amount'] = budget_breakdown['Amount'].round(2)
            budget_breakdown['Percentage'] = budget_breakdown['Percentage'].round(1)
            budget_breakdown['Monthly Average'] = budget_breakdown['Monthly Average'].round(2)
            return budget_breakdown
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return pd.DataFrame()

# Main app
def main():
    # Apply theme
    apply_theme()
    
    # Title and theme toggle with Mario Kart styling
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">💰 SUPER BUDGET BROS</h1>
            <p style="font-size: 1.2rem; color: #FFD700;">🏎️ Race to Financial Freedom! 🏎️</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Character selection with enhanced styling
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h2 style="color: #FFD700; text-shadow: 2px 2px 0px #8B4513;">👥 SELECT YOUR RACER</h2>
        </div>
    """, unsafe_allow_html=True)
    
    selected_character = st.sidebar.selectbox(
        "Choose your character",
        options=list(CHARACTERS.keys()),
        format_func=lambda x: f"{CHARACTERS[x]['icon']} {x}",
        key="character_select"
    )
    
    # Update selected character
    st.session_state.selected_character = selected_character
    
    # Display character info with enhanced styling and fallback
    character = CHARACTERS[selected_character]
    st.sidebar.markdown(f"""
        <div class='track-selector' style="text-align: center;">
            <h3 style='color: {character['color']}; text-shadow: 2px 2px 0px #8B4513;'>{character['icon']} {selected_character}</h3>
            <div style="font-size: 5rem; margin: 1rem 0;">{character['fallback']}</div>
            <p style="font-size: 0.8rem;">{character['description']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Track selection with enhanced styling
    st.sidebar.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h2 style="color: #FFD700; text-shadow: 2px 2px 0px #8B4513;">🏁 SELECT YOUR TRACK</h2>
        </div>
    """, unsafe_allow_html=True)
    
    selected_track = st.sidebar.selectbox(
        "Choose your track",
        options=character['tracks'],
        format_func=lambda x: f"🏎️ {x}",
        key="track_select"
    )
    
    # Update selected track
    st.session_state.selected_track = selected_track
    
    # Navigation tabs with enhanced Mario Kart styling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 BUDGET ANALYZER",
        "💰 FINANCIAL TOOLKIT",
        "📄 REPORTS GENERATOR",
        "🎮 BUDGET GAME"
    ])
    
    # Budget Analyzer Tab
    with tab1:
        st.header("📊 Budget Analyzer")
        
        # Data source selection
        data_source = st.radio("Select Data Source", ["Upload CSV", "Use Sample Data"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your financial data (CSV)", type=['csv'])
            if uploaded_file is not None:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.df['Date'] = pd.to_datetime(st.session_state.df['Date'])
                st.success("Data uploaded successfully! 🍄")
                
                if st.session_state.df is not None:
                    df = st.session_state.df
                    
                    # User Preferences Section
                    st.subheader("🎯 Set Your Financial Preferences")
                    
                    # Risk Tolerance Assessment
                    st.write("### Risk Tolerance Assessment")
                    risk_questions = {
                        "How do you feel about investment risk?": {
                            "Very uncomfortable": 1,
                            "Somewhat uncomfortable": 2,
                            "Neutral": 3,
                            "Comfortable": 4,
                            "Very comfortable": 5
                        },
                        "How would you react to a 20% market drop?": {
                            "Sell everything": 1,
                            "Sell some": 2,
                            "Hold": 3,
                            "Buy more": 4,
                            "Buy aggressively": 5
                        },
                        "Your investment time horizon is:": {
                            "Less than 1 year": 1,
                            "1-3 years": 2,
                            "3-5 years": 3,
                            "5-10 years": 4,
                            "More than 10 years": 5
                        }
                    }
                    
                    risk_answers = {}
                    for question, options in risk_questions.items():
                        risk_answers[question] = st.select_slider(question, options=list(options.keys()))
                    
                    if st.button("Assess Risk Tolerance"):
                        risk_level, risk_description = assess_risk_tolerance(risk_answers)
                        st.session_state.user_preferences['risk_tolerance'] = risk_level
                        st.success(f"Your risk tolerance is: {risk_level}")
                        st.info(risk_description)
                    
                    # Savings Goal
                    st.write("### Set Your Savings Goal")
                    monthly_income = df[df['Type'] == 'Income']['Amount'].sum() / 6
                    savings_goal = st.slider(
                        "Target Monthly Savings Rate (%)",
                        min_value=0,
                        max_value=100,
                        value=20,
                        help="Recommended: 20% of monthly income"
                    )
                    st.session_state.user_preferences['savings_goal'] = savings_goal
                    
                    # Investment Preferences
                    st.write("### Investment Preferences")
                    investment_options = {
                        "Stocks": "Higher risk, higher potential returns",
                        "Bonds": "Lower risk, stable returns",
                        "Real Estate": "Medium risk, long-term growth",
                        "Cryptocurrency": "Very high risk, volatile returns",
                        "Savings Account": "Lowest risk, guaranteed returns"
                    }
                    
                    selected_investments = st.multiselect(
                        "Select your preferred investment types",
                        options=list(investment_options.keys()),
                        help="Choose multiple options that interest you"
                    )
                    st.session_state.user_preferences['investment_preference'] = selected_investments
                    
                    # Generate Recommendations
                    if st.button("Generate Recommendations"):
                        recommendations = generate_recommendations(df, st.session_state.user_preferences)
                        st.subheader("📈 Personalized Recommendations")
                        for rec in recommendations:
                            st.write(rec)
                    
                    # Summary Dashboard
                    st.subheader("📊 Summary Dashboard")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_income = df[df['Type'] == 'Income']['Amount'].sum()
                        st.metric("Total Income", f"${total_income:,.2f}")
                    with col2:
                        total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
                        st.metric("Total Expenses", f"${total_expenses:,.2f}")
                    with col3:
                        net_savings = total_income - total_expenses
                        st.metric("Net Savings", f"${net_savings:,.2f}")
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        expense_by_category = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
                        fig_pie = px.pie(
                            values=expense_by_category.values,
                            names=expense_by_category.index,
                            title='Expenses by Category'
                        )
                        fig_pie = style_visualization(fig_pie, character)
                        st.plotly_chart(fig_pie, use_container_width=True, key="budget_analyzer_pie_chart")
                        
                        # Sample data preview (only for sample data)
                        if data_source == "Use Sample Data":
                            st.subheader("📊 Sample Data Preview")
                            st.dataframe(df.head(10))
                    
                    with col2:
                        # Line chart with predictions
                        monthly_totals = df.groupby(['Date', 'Type'])['Amount'].sum().unstack().fillna(0)
                        fig_line = go.Figure()
                        
                        fig_line.add_trace(go.Scatter(
                            x=monthly_totals.index,
                            y=monthly_totals['Income'],
                            name='Income',
                            line=dict(color=THEME['light' if not st.session_state.dark_mode else 'dark']['success'])
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=monthly_totals.index,
                            y=monthly_totals['Expense'],
                            name='Expense',
                            line=dict(color=THEME['light' if not st.session_state.dark_mode else 'dark']['danger'])
                        ))
                        
                        fig_line = style_visualization(fig_line, character)
                        st.plotly_chart(fig_line, use_container_width=True, key="budget_analyzer_line_chart")
                    
                    # Anomaly Detection
                    st.subheader("⚠️ Anomaly Detection")
                    anomalies = detect_anomalies(df)
                    if not anomalies.empty:
                        st.write("Unusual spending patterns detected:")
                        st.dataframe(anomalies)
                    else:
                        st.info("No unusual spending patterns detected in your data.")
                    
                    # Monthly Net Savings Trend
                    st.subheader("📊 Monthly Net Savings Trend")
                    monthly_data = df.groupby(pd.Grouper(key='Date', freq='ME')).agg({
                        'Amount': lambda x: x[df['Type'] == 'Income'].sum() - x[df['Type'] == 'Expense'].sum()
                    }).reset_index()
                    
                    fig_trend = px.line(
                        monthly_data,
                        x='Date',
                        y='Amount',
                        title='Monthly Net Savings Trend',
                        labels={'Amount': 'Net Savings ($)'}
                    )
                    fig_trend.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Net Savings ($)',
                        showlegend=False
                    )
                    st.plotly_chart(fig_trend, use_container_width=True, key="monthly_savings_trend_chart")
        
        else:  # Sample Data
            st.session_state.df = generate_sample_data()
            st.session_state.df['Date'] = pd.to_datetime(st.session_state.df['Date'])
            st.info("Using sample data for demonstration 🏰")
            
            if st.session_state.df is not None:
                df = st.session_state.df
                
                # Show sample data preview
                st.subheader("📊 Sample Data Preview")
                st.dataframe(df.head(10))
                
                # Rest of the sample data analysis...
    
    # Financial Toolkit Tab
    with tab2:
        st.header("💰 Financial Toolkit")
        
        # Expense Optimizer
        st.subheader("📉 Expense Optimizer")
        if st.session_state.df is not None:
            df = st.session_state.df
            expense_by_category = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
            avg_expense = expense_by_category.mean()
            
            st.write("Categories needing optimization:")
            for category, amount in expense_by_category.items():
                if amount > avg_expense:
                    st.warning(f"⚠️ {category}: ${amount:,.2f} (${amount - avg_expense:,.2f} above average)")
        
        # Loan Eligibility Checker
        st.subheader("🏦 Loan Eligibility Checker")
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            expenses = st.number_input("Annual Expenses ($)", min_value=0, value=30000)
        
        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 700)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
        
        if st.button("Check Eligibility"):
            eligible, probability = check_loan_eligibility(income, expenses, credit_score, loan_amount)
            if eligible:
                st.success(f"✅ Loan approved! (Probability: {probability*100:.1f}%)")
            else:
                st.error(f"❌ Loan not approved (Probability: {probability*100:.1f}%)")
        
        # Investment Return Calculator
        st.subheader("📈 Investment Return Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            initial_investment = st.number_input("Initial Investment ($)", min_value=0, value=10000)
            years = st.slider("Investment Duration (Years)", 1, 30, 10)
        
        with col2:
            annual_return = st.slider("Expected Annual Return (%)", 1, 20, 7)
            monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=100)
        
        if st.button("Calculate Returns"):
            monthly_rate = annual_return / 12 / 100
            months = years * 12
            future_value = initial_investment * (1 + monthly_rate) ** months
            for i in range(months):
                future_value += monthly_contribution * (1 + monthly_rate) ** (months - i)
            
            st.success(f"Future Value: ${future_value:,.2f}")
    
    # Reports Tab
    with tab3:
        st.header("📊 Reports")
        
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            
            # Report type selection
            report_type = st.radio(
                "Select Report Type",
                ["Summary Report", "Detailed Transactions", "Budget Analysis"]
            )
            
            # Generate and display report
            report_df = generate_report(df, report_type, st.session_state.user_preferences)
            
            if report_type == "Summary Report":
                st.subheader("📊 Summary Report")
                for _, row in report_df.iterrows():
                    if 'Metric' in row and row['Metric'] == 'Savings Rate':
                        st.metric(row['Metric'], f"{row['Amount']:.1f}%")
                    elif 'Metric' in row:
                        st.metric(row['Metric'], f"${row['Amount']:,.2f}")
            
            elif report_type == "Detailed Transactions":
                st.subheader("📝 Detailed Transactions")
                display_df = report_df.copy()
                if 'Date' in display_df.columns:
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df, use_container_width=True)
            
            else:  # Budget Analysis
                st.subheader("💰 Budget Analysis")
                display_df = report_df.copy()
                display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
                display_df['Monthly Average'] = display_df['Monthly Average'].apply(lambda x: f"${x:,.2f}")
                display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, use_container_width=True)
            
            # Download report
            if st.button("📥 Download Report"):
                csv = report_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{report_type.lower().replace(" ", "_")}_report.csv">Download {report_type}</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # Financial Education Tab
    with tab4:
        st.header("📚 Financial Education")
        
        # Game state
        if 'game_state' not in st.session_state:
            st.session_state.game_state = {
                'financial_health': 100,
                'savings_rate': 50,
                'investment_balance': 100,
                'credit_score': 0
            }
        
        # Financial scenario
        st.subheader("📊 Current Financial Scenario")
        scenario = st.selectbox(
            "Choose a financial scenario",
            [
                "You have $10,000 to invest. What do you do?",
                "Your emergency fund is low. How do you handle it?",
                "You received a bonus. How do you use it?",
                "You want to save for a major purchase. What's your plan?"
            ]
        )
        
        # Financial choices
        st.subheader("🎯 Make Your Decision")
        if scenario == "You have $10,000 to invest. What do you do?":
            choice = st.radio(
                "Select your action",
                [
                    "Save 50% and invest the rest in diversified funds",
                    "Invest 30% in stocks, 20% in bonds, save the rest",
                    "Pay off high-interest debt and invest the remainder",
                    "Create an emergency fund and invest the rest"
                ]
            )
        elif scenario == "Your emergency fund is low. How do you handle it?":
            choice = st.radio(
                "Select your action",
                [
                    "Build emergency fund immediately",
                    "Reduce expenses and save gradually",
                    "Use credit card as backup",
                    "Delay building emergency fund"
                ]
            )
        elif scenario == "You received a bonus. How do you use it?":
            choice = st.radio(
                "Select your action",
                [
                    "Invest 70% and save 30%",
                    "Pay off high-interest debt",
                    "Add to emergency fund",
                    "Use for regular expenses"
                ]
            )
        else:  # Major purchase scenario
            choice = st.radio(
                "Select your action",
                [
                    "Create a dedicated savings plan",
                    "Use investment returns and save the rest",
                    "Take a loan and pay it off later",
                    "Postpone until you have enough saved"
                ]
            )
        
        # Financial outcome
        if st.button("Submit Decision"):
            if "save" in choice.lower() or "invest" in choice.lower():
                st.session_state.game_state['financial_health'] = min(100, st.session_state.game_state['financial_health'] + 20)
                st.session_state.game_state['savings_rate'] = min(100, st.session_state.game_state['savings_rate'] + 10)
                st.session_state.game_state['credit_score'] += 50
                st.success("Great decision! Your financial health improved! 🎉")
            elif "debt" in choice.lower() or "pay off" in choice.lower():
                st.session_state.game_state['financial_health'] = min(100, st.session_state.game_state['financial_health'] + 10)
                st.session_state.game_state['investment_balance'] = min(100, st.session_state.game_state['investment_balance'] + 20)
                st.session_state.game_state['credit_score'] += 30
                st.success("Good choice! Your financial position is stronger! 🎯")
            else:
                st.session_state.game_state['financial_health'] = max(0, st.session_state.game_state['financial_health'] - 10)
                st.session_state.game_state['savings_rate'] = max(0, st.session_state.game_state['savings_rate'] - 5)
                st.session_state.game_state['investment_balance'] = max(0, st.session_state.game_state['investment_balance'] - 10)
                st.warning("Be careful! Your financial health took a hit! ⚠️")
        
        # Display financial stats
        st.subheader("📊 Financial Health Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Financial Health", f"{st.session_state.game_state['financial_health']}%")
        with col2:
            st.metric("Savings Rate", f"{st.session_state.game_state['savings_rate']}%")
        with col3:
            st.metric("Investment Balance", f"${st.session_state.game_state['investment_balance']}")
        with col4:
            st.metric("Credit Score", f"{st.session_state.game_state['credit_score']}")

if __name__ == "__main__":
    main()