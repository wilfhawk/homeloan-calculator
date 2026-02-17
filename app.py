
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Home Loan Calculator Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
    /* Mobile-friendly adjustments */
    @media (max-width: 768px) {
        .stPlotlyChart {
            height: 400px !important;
        }
        .row-widget.stButton button {
            width: 100%;
        }
    }
    
    /* Better chart rendering */
    .js-plotly-plot {
        width: 100% !important;
    }
    
    /* Improve metric cards on mobile */
    [data-testid="stMetricValue"] {
        font-size: clamp(1.2rem, 3vw, 2rem);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def calculate_pmi_monthly(loan_amount, home_price):
    """Calculate monthly PMI if down payment < 20%"""
    ltv = (loan_amount / home_price) * 100
    
    if ltv <= 80:
        return 0
    
    # PMI rates vary by LTV
    if ltv > 95:
        pmi_rate = 0.015  # 1.5%
    elif ltv > 90:
        pmi_rate = 0.010  # 1.0%
    elif ltv > 85:
        pmi_rate = 0.0075  # 0.75%
    else:
        pmi_rate = 0.005  # 0.5%
    
    annual_pmi = loan_amount * pmi_rate
    return annual_pmi / 12

def calculate_monthly_mortgage(loan_amount, annual_rate, years):
    """Calculate monthly mortgage payment (P&I only)"""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    
    if monthly_rate > 0:
        payment = loan_amount * (
            monthly_rate * (1 + monthly_rate)**num_payments
        ) / ((1 + monthly_rate)**num_payments - 1)
    else:
        payment = loan_amount / num_payments
    
    return payment

def calculate_arm_payment(loan_amount, current_rate, years_remaining):
    """Calculate ARM payment with current rate"""
    return calculate_monthly_mortgage(loan_amount, current_rate, years_remaining)

def generate_arm_schedule(loan_amount, initial_rate, loan_term, fixed_period, rate_adjustments, caps):
    """Generate ARM amortization with rate adjustments"""
    schedule = []
    balance = loan_amount
    current_rate = initial_rate
    months_elapsed = 0
    
    # Calculate total months and monthly payment
    total_months = loan_term * 12
    monthly_rate = current_rate / 100 / 12
    
    for month in range(1, total_months + 1):
        current_year = (month - 1) // 12 + 1
        month_in_year = (month - 1) % 12 + 1
        
        # Check if rate should adjust this month
        if current_year > fixed_period:
            months_since_fixed = (current_year - fixed_period - 1) * 12 + month_in_year
            adjustment_period_months = rate_adjustments['frequency'] * 12
            
            if months_since_fixed > 0 and months_since_fixed % adjustment_period_months == 1:
                # Apply rate adjustment
                rate_change = rate_adjustments['expected_change']
                rate_change = max(min(rate_change, caps['periodic']), -caps['periodic'])
                new_rate = current_rate + rate_change
                new_rate = max(min(new_rate, initial_rate + caps['lifetime_max']), max(0, initial_rate - 2.0))
                current_rate = new_rate
                monthly_rate = current_rate / 100 / 12
        
        # Recalculate payment at rate changes
        if month_in_year == 1 or (month > 1 and current_rate != schedule[-1]['Rate'] if schedule else False):
            months_remaining = total_months - month + 1
            if balance > 0 and months_remaining > 0:
                if monthly_rate > 0:
                    monthly_payment = balance * (
                        monthly_rate * (1 + monthly_rate)**months_remaining
                    ) / ((1 + monthly_rate)**months_remaining - 1)
                else:
                    monthly_payment = balance / months_remaining
            else:
                monthly_payment = 0
        
        if balance <= 0:
            break
            
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment
        
        # Store yearly summary
        if month_in_year == 12 or month == total_months:
            if not schedule or schedule[-1]['Year'] != current_year:
                schedule.append({
                    'Year': current_year,
                    'Rate': round(current_rate, 2),
                    'Monthly_Payment': round(monthly_payment, 2),
                    'Principal': round(principal_payment, 2),
                    'Interest': round(interest_payment, 2),
                    'Balance': round(max(0, balance), 2)
                })
            else:
                # Update existing year with cumulative values
                schedule[-1]['Principal'] += principal_payment
                schedule[-1]['Interest'] += interest_payment
                schedule[-1]['Balance'] = round(max(0, balance), 2)
                schedule[-1]['Monthly_Payment'] = round(monthly_payment, 2)
    
    return pd.DataFrame(schedule)

def calculate_refinance_savings(current_balance, current_rate, years_left, new_rate, new_term, refi_costs):
    """Calculate refinancing savings"""
    # Current monthly payment
    current_payment = calculate_monthly_mortgage(current_balance, current_rate, years_left)
    
    # New monthly payment after refinance
    new_payment = calculate_monthly_mortgage(current_balance, new_rate, new_term)
    
    # Monthly savings
    monthly_savings = current_payment - new_payment
    
    # Break-even months (how long to recover refinance costs)
    break_even_months = refi_costs / monthly_savings if monthly_savings > 0 else float('inf')
    
    # Lifetime savings (comparing same time period)
    comparison_period = min(years_left, new_term)
    total_current = current_payment * comparison_period * 12
    total_new = new_payment * comparison_period * 12 + refi_costs
    lifetime_savings = total_current - total_new
    
    return {
        'current_payment': current_payment,
        'new_payment': new_payment,
        'monthly_savings': monthly_savings,
        'break_even_months': break_even_months,
        'lifetime_savings': lifetime_savings
    }

def calculate_selling_costs(home_value, original_price, years_owned):
    """Calculate costs and taxes when selling"""
    # Agent commission (typically 5-6%)
    agent_commission = home_value * 0.06
    
    # Capital gains
    capital_gain = home_value - original_price
    
    # Capital gains exclusion ($250k single, $500k married)
    # Simplified: assume married for this calculation
    excluded_gain = 500000
    taxable_gain = max(0, capital_gain - excluded_gain)
    
    # Long-term capital gains tax - use 15% rate for long-term (most common scenario)
    # For homes held > 1 year, gains are long-term and taxed at favorable rates
    if years_owned >= 1:
        capital_gains_tax = taxable_gain * 0.15  # Typical for long-term holds
    else:
        capital_gains_tax = taxable_gain * 0.37  # Taxed as ordinary income if < 1 year
    
    # Other closing costs when selling
    title_fees = 500
    transfer_taxes = home_value * 0.001
    other_fees = 1000
    
    total_costs = agent_commission + capital_gains_tax + title_fees + transfer_taxes + other_fees
    
    return {
        'agent_commission': agent_commission,
        'capital_gain': capital_gain,
        'taxable_gain': taxable_gain,
        'capital_gains_tax': capital_gains_tax,
        'title_fees': title_fees,
        'transfer_taxes': transfer_taxes,
        'other_fees': other_fees,
        'total_selling_costs': total_costs,
        'net_proceeds': home_value - total_costs
    }

def monte_carlo_simulation(base_params, n_simulations=1000):
    """Run Monte Carlo simulation with variable rates and appreciation"""
    # Note: Removed seed for true randomness. Uncomment line below for reproducible results:
    # np.random.seed(42)  # For reproducibility
    
    results = []
    
    for _ in range(n_simulations):
        # Vary key parameters
        appreciation = np.random.normal(base_params['home_appreciation'], 2.0)
        appreciation = max(-5, min(15, appreciation))  # Constrain to reasonable range
        
        rent_increase = np.random.normal(base_params['rent_increase'], 1.5)
        rent_increase = max(0, min(10, rent_increase))
        
        # Calculate outcome for this simulation
        current_home_value = base_params['home_price']
        cumulative_own = base_params['down_payment']
        
        for year in range(base_params['analysis_years']):
            # Home appreciation
            current_home_value *= (1 + appreciation / 100)
            
            # Annual ownership cost (simplified)
            yearly_cost = base_params['annual_ownership_cost']
            cumulative_own += yearly_cost
        
        # Calculate equity
        equity = current_home_value - base_params['remaining_loan']
        net_cost = cumulative_own - equity
        
        results.append({
            'appreciation_rate': appreciation,
            'final_home_value': current_home_value,
            'total_equity': equity,
            'net_cost': net_cost
        })
    
    return pd.DataFrame(results)

def adjust_for_inflation(amount, years, inflation_rate):
    """Adjust future amount to present value"""
    return amount / ((1 + inflation_rate / 100) ** years)

def generate_amortization_schedule(loan_amount, annual_rate, years, start_year=1):
    """Generate complete amortization schedule"""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    monthly_payment = calculate_monthly_mortgage(loan_amount, annual_rate, years)
    
    schedule = []
    balance = loan_amount
    
    for month in range(1, num_payments + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment
        
        schedule.append({
            'Month': month,
            'Year': start_year + (month - 1) // 12,
            'Payment': monthly_payment,
            'Principal': principal_payment,
            'Interest': interest_payment,
            'Balance': max(0, balance)
        })
    
    return pd.DataFrame(schedule)

def calculate_tax_savings(mortgage_interest, property_tax, income, filing_status):
    """Calculate annual tax savings from mortgage interest and property tax deductions"""
    standard_deductions = {
        'Single': 14600,
        'Married Filing Jointly': 29200,
        'Head of Household': 21900
    }
    
    # Tax brackets defined as (upper_limit, rate) - must be in ascending order
    tax_brackets = {
        'Single': [(11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24), (243725, 0.32), (609350, 0.35), (float('inf'), 0.37)],
        'Married Filing Jointly': [(23200, 0.10), (94300, 0.12), (201050, 0.22), (383900, 0.24), (487450, 0.32), (731200, 0.35), (float('inf'), 0.37)],
        'Head of Household': [(16550, 0.10), (63100, 0.12), (100500, 0.22), (191950, 0.24), (243700, 0.32), (609350, 0.35), (float('inf'), 0.37)]
    }
    
    standard_deduction = standard_deductions[filing_status]
    itemized_deduction = mortgage_interest + property_tax
    
    if itemized_deduction <= standard_deduction:
        return 0
    
    extra_deduction = itemized_deduction - standard_deduction
    
    # Find the marginal tax rate for this income level
    brackets = tax_brackets[filing_status]
    marginal_rate = 0.37  # Default to highest rate
    for limit, rate in brackets:
        if income <= limit:
            marginal_rate = rate
            break
    
    # Tax savings = extra deduction * marginal tax rate
    tax_savings = extra_deduction * marginal_rate
    return tax_savings

def calculate_closing_costs(home_price, loan_amount):
    """Estimate closing costs"""
    origination_fee = loan_amount * 0.01
    appraisal = 500
    title_insurance = home_price * 0.005
    inspection = 500
    recording_fees = 200
    other_fees = 1000
    
    total = origination_fee + appraisal + title_insurance + inspection + recording_fees + other_fees
    
    return {
        'origination_fee': origination_fee,
        'appraisal': appraisal,
        'title_insurance': title_insurance,
        'inspection': inspection,
        'recording_fees': recording_fees,
        'other_fees': other_fees,
        'total': total
    }

def calculate_break_even_point(ownership_costs_yearly, rental_costs_yearly, home_equity_yearly):
    """Find the year when buying becomes cheaper than renting"""
    cumulative_own = 0
    cumulative_rent = 0
    
    for year in range(len(ownership_costs_yearly)):
        cumulative_own += ownership_costs_yearly[year]
        cumulative_rent += rental_costs_yearly[year]
        
        net_ownership = cumulative_own - home_equity_yearly[year]
        
        if net_ownership < cumulative_rent:
            return year + 1
    
    return None

# ==================== MAIN APP ====================

st.title("🏠 Professional Home Loan Calculator")
st.markdown("""
Comprehensive financial analysis tool with ARM mortgages, PMI, refinancing, 
Monte Carlo simulations, and inflation adjustments.
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Main Analysis", 
    "🔢 Amortization", 
    "🔄 Refinance Calculator",
    "🎲 Monte Carlo",
    "⚖️ Compare Scenarios", 
    "💾 Export"
])

# ==================== SIDEBAR: INPUTS ====================

with st.sidebar:
    st.header("Configuration")
    
    scenario_name = st.text_input("Scenario Name", value="Default Scenario")
    
    # Mortgage type selector
    st.subheader("🏦 Loan Type")
    loan_type = st.radio(
        "Mortgage Type",
        options=["Fixed Rate", "ARM (Adjustable Rate)"],
        help="Choose between fixed rate or adjustable rate mortgage"
    )
    
    st.divider()
    st.subheader("🏡 Home Purchase")
    
    home_price = st.number_input(
        "Home Price ($)",
        min_value=50000,
        max_value=5000000,
        value=400000,
        step=10000
    )
    
    down_payment_pct = st.slider(
        "Down Payment (%)",
        min_value=0,
        max_value=50,
        value=20
    )
    
    if loan_type == "Fixed Rate":
        interest_rate = st.number_input(
            "Annual Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=6.5,
            step=0.1
        )
        
        loan_term = st.selectbox(
            "Loan Term (years)",
            options=[15, 20, 30],
            index=2
        )
    else:
        # ARM specific inputs
        initial_rate = st.number_input(
            "Initial Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
            help="Starting rate for ARM"
        )
        
        loan_term = st.selectbox(
            "Loan Term (years)",
            options=[15, 20, 30],
            index=2
        )
        
        fixed_period = st.selectbox(
            "Fixed Period (years)",
            options=[3, 5, 7, 10],
            index=1,
            help="How long rate stays fixed (e.g., 5/1 ARM = 5 years)"
        )
        
        adjustment_frequency = st.selectbox(
            "Rate Adjustment Frequency",
            options=[1, 2, 3],
            index=0,
            help="How often rate adjusts after fixed period (years)"
        )
        
        expected_rate_change = st.slider(
            "Expected Rate Change per Adjustment (%)",
            min_value=-2.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Expected change each adjustment period"
        )
        
        periodic_cap = st.slider(
            "Periodic Cap (%)",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.5,
            help="Maximum rate change per adjustment"
        )
        
        lifetime_cap = st.slider(
            "Lifetime Cap (%)",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Maximum total rate increase over loan life"
        )
    
    include_closing = st.checkbox("Include Closing Costs", value=True)
    
    property_tax_annual = st.number_input(
        "Annual Property Tax ($)",
        min_value=0,
        max_value=50000,
        value=5000,
        step=500
    )
    
    home_insurance_annual = st.number_input(
        "Annual Home Insurance ($)",
        min_value=0,
        max_value=10000,
        value=1200,
        step=100
    )
    
    hoa_monthly = st.number_input(
        "Monthly HOA Fees ($)",
        min_value=0,
        max_value=2000,
        value=0,
        step=50
    )
    
    maintenance_pct = st.slider(
        "Annual Maintenance (% of home value)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
    
    st.divider()
    st.subheader("🏢 Rental")
    
    monthly_rent = st.number_input(
        "Monthly Rent ($)",
        min_value=500,
        max_value=20000,
        value=2500,
        step=100
    )
    
    rent_increase_annual = st.slider(
        "Expected Annual Rent Increase (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    renters_insurance_annual = st.number_input(
        "Annual Renter's Insurance ($)",
        min_value=0,
        max_value=1000,
        value=200,
        step=50
    )
    
    st.divider()
    st.subheader("💰 Financial Parameters")
    
    annual_income = st.number_input(
        "Annual Household Income ($)",
        min_value=0,
        max_value=1000000,
        value=100000,
        step=5000
    )
    
    filing_status = st.selectbox(
        "Tax Filing Status",
        options=['Single', 'Married Filing Jointly', 'Head of Household'],
        index=1
    )
    
    inflation_rate = st.slider(
        "Expected Inflation Rate (%/year)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.5,
        help="Used to adjust future values to present value"
    )
    
    st.divider()
    st.subheader("📊 Analysis")
    
    analysis_years = st.slider(
        "Analysis Period (years)",
        min_value=1,
        max_value=30,
        value=10
    )
    
    home_appreciation = st.slider(
        "Expected Home Appreciation (%/year)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    investment_return = st.slider(
        "Alternative Investment Return (%/year)",
        min_value=0.0,
        max_value=15.0,
        value=7.0,
        step=0.5
    )
    
    plan_to_sell = st.checkbox("Plan to Sell at End of Analysis Period", value=False)
    
    # Save scenario
    if st.button("💾 Save This Scenario", use_container_width=True):
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = []
        
        scenario = {
            'name': scenario_name,
            'loan_type': loan_type,
            'home_price': home_price,
            'down_payment_pct': down_payment_pct,
            'monthly_rent': monthly_rent,
            'analysis_years': analysis_years
        }
        
        st.session_state.scenarios.append(scenario)
        st.success(f"✅ Saved '{scenario_name}'!")

# ==================== CALCULATIONS ====================

down_payment = home_price * (down_payment_pct / 100)
loan_amount = home_price - down_payment

# PMI calculation
monthly_pmi = calculate_pmi_monthly(loan_amount, home_price)
show_pmi_warning = monthly_pmi > 0

# Closing costs
closing_costs = calculate_closing_costs(home_price, loan_amount) if include_closing else None

# Calculate mortgage based on type
if loan_type == "Fixed Rate":
    monthly_mortgage = calculate_monthly_mortgage(loan_amount, interest_rate, loan_term)
    amortization = generate_amortization_schedule(loan_amount, interest_rate, loan_term)
else:
    # ARM calculations
    arm_caps = {
        'periodic': periodic_cap,
        'lifetime_max': lifetime_cap
    }
    arm_adjustments = {
        'frequency': adjustment_frequency,
        'expected_change': expected_rate_change
    }
    amortization = generate_arm_schedule(
        loan_amount, 
        initial_rate, 
        loan_term, 
        fixed_period,
        arm_adjustments,
        arm_caps
    )
    # Get initial payment (first year)
    if not amortization.empty:
        monthly_mortgage = amortization.iloc[0]['Monthly_Payment']
    else:
        monthly_mortgage = calculate_monthly_mortgage(loan_amount, initial_rate, loan_term)
    interest_rate = initial_rate  # For display purposes

# Monthly costs
monthly_property_tax = property_tax_annual / 12
monthly_insurance = home_insurance_annual / 12
monthly_maintenance = (home_price * (maintenance_pct / 100)) / 12

total_monthly_ownership = (
    monthly_mortgage + 
    monthly_property_tax + 
    monthly_insurance + 
    hoa_monthly + 
    monthly_maintenance +
    monthly_pmi
)

monthly_renters_insurance = renters_insurance_annual / 12
total_monthly_rental = monthly_rent + monthly_renters_insurance

# Multi-year analysis
years_list = []
ownership_cumulative = []
rental_cumulative = []
home_equity = []
tax_savings_cumulative = []
ownership_costs_yearly = []
rental_costs_yearly = []
home_equity_yearly = []
ownership_real_value = []  # Inflation-adjusted
rental_real_value = []

current_home_value = home_price
current_rent = monthly_rent
cumulative_own = down_payment
if include_closing and closing_costs:
    cumulative_own += closing_costs['total']
cumulative_rent = 0
remaining_principal = loan_amount
cumulative_tax_savings = 0

for year in range(1, analysis_years + 1):
    # Get this year's data
    if loan_type == "Fixed Rate":
        year_schedule = amortization[amortization['Year'] == year]
        yearly_interest = year_schedule['Interest'].sum()
        yearly_principal = year_schedule['Principal'].sum()
        year_mortgage_payment = monthly_mortgage * 12
    else:
        year_data = amortization[amortization['Year'] == year]
        if not year_data.empty:
            yearly_interest = year_data['Interest'].iloc[0]
            yearly_principal = year_data['Principal'].iloc[0]
            year_mortgage_payment = year_data['Monthly_Payment'].iloc[0] * 12
        else:
            yearly_interest = 0
            yearly_principal = 0
            year_mortgage_payment = 0
    
    # PMI drops off when LTV reaches 78%
    current_pmi = monthly_pmi * 12
    current_ltv = (remaining_principal / current_home_value) * 100
    if current_ltv <= 78:
        current_pmi = 0
    
    # Tax savings
    tax_savings = calculate_tax_savings(
        yearly_interest,
        property_tax_annual,
        annual_income,
        filing_status
    )
    cumulative_tax_savings += tax_savings
    
    # Ownership costs
    yearly_ownership = (
        year_mortgage_payment +
        property_tax_annual +
        home_insurance_annual +
        (hoa_monthly * 12) +
        (monthly_maintenance * 12) +
        current_pmi -
        tax_savings
    )
    cumulative_own += yearly_ownership
    ownership_costs_yearly.append(yearly_ownership)
    
    # Rental costs
    yearly_rental = (current_rent + monthly_renters_insurance) * 12
    cumulative_rent += yearly_rental
    rental_costs_yearly.append(yearly_rental)
    
    # Home appreciation
    current_home_value *= (1 + home_appreciation / 100)
    
    # Equity
    remaining_principal -= yearly_principal
    equity = current_home_value - max(0, remaining_principal)
    home_equity_yearly.append(equity)
    
    # Inflation adjustment
    real_own = adjust_for_inflation(cumulative_own, year, inflation_rate)
    real_rent = adjust_for_inflation(cumulative_rent, year, inflation_rate)
    
    # Store
    years_list.append(year)
    ownership_cumulative.append(cumulative_own)
    rental_cumulative.append(cumulative_rent)
    home_equity.append(equity)
    tax_savings_cumulative.append(cumulative_tax_savings)
    ownership_real_value.append(real_own)
    rental_real_value.append(real_rent)
    
    # Next year rent
    current_rent *= (1 + rent_increase_annual / 100)

# Selling costs if applicable
selling_costs = None
if plan_to_sell:
    selling_costs = calculate_selling_costs(
        current_home_value,
        home_price,
        analysis_years
    )

# Break-even
break_even_year = calculate_break_even_point(
    ownership_costs_yearly,
    rental_costs_yearly,
    home_equity_yearly
)

# ==================== TAB 1: MAIN ANALYSIS ====================

with tab1:
    # PMI Warning
    if show_pmi_warning:
        st.warning(f"⚠️ **PMI Required:** Your down payment is less than 20%. Monthly PMI: ${monthly_pmi:,.2f}")
        ltv = (loan_amount / home_price) * 100
        st.caption(f"Current LTV: {ltv:.1f}%. PMI will drop off when you reach 78% LTV (approximately ${home_price * 0.78:,.0f} loan balance)")
    
    # Quick metrics
    st.subheader("💰 Monthly Cost Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Ownership", f"${total_monthly_ownership:,.2f}")
    
    with col2:
        st.metric("Monthly Rental", f"${total_monthly_rental:,.2f}")
    
    with col3:
        difference = total_monthly_ownership - total_monthly_rental
        st.metric("Difference", f"${abs(difference):,.2f}", 
                 delta="Buying costs more" if difference > 0 else "Renting costs more")
    
    with col4:
        if loan_type == "ARM":
            st.metric("Loan Type", "ARM", help=f"{fixed_period}-year fixed, then adjustable")
        else:
            st.metric("Loan Type", "Fixed", help=f"{loan_term}-year fixed rate")
    
    # Cost breakdown with PMI
    st.divider()
    st.subheader("📋 Monthly Cost Breakdown")
    
    breakdown_col1, breakdown_col2 = st.columns(2)
    
    with breakdown_col1:
        st.markdown("**Ownership Costs:**")
        st.write(f"- Mortgage Payment (P&I): ${monthly_mortgage:,.2f}")
        if monthly_pmi > 0:
            st.write(f"- PMI: ${monthly_pmi:,.2f}")
        st.write(f"- Property Tax: ${monthly_property_tax:,.2f}")
        st.write(f"- Home Insurance: ${monthly_insurance:,.2f}")
        st.write(f"- HOA Fees: ${hoa_monthly:,.2f}")
        st.write(f"- Maintenance: ${monthly_maintenance:,.2f}")
        st.write(f"- **Down Payment: ${down_payment:,.2f}** (one-time)")
        if include_closing and closing_costs:
            st.write(f"- **Closing Costs: ${closing_costs['total']:,.2f}** (one-time)")
    
    with breakdown_col2:
        st.markdown("**Rental Costs:**")
        st.write(f"- Monthly Rent: ${monthly_rent:,.2f}")
        st.write(f"- Renter's Insurance: ${monthly_renters_insurance:,.2f}")
    
    # Tax benefits
    st.divider()
    st.subheader("🧾 Tax Benefits")
    
    if loan_type == "Fixed Rate":
        first_year_interest = amortization[amortization['Year'] == 1]['Interest'].sum()
    else:
        first_year_interest = amortization.iloc[0]['Interest']
    
    first_year_tax_savings = calculate_tax_savings(
        first_year_interest,
        property_tax_annual,
        annual_income,
        filing_status
    )
    
    tax_col1, tax_col2, tax_col3 = st.columns(3)
    
    with tax_col1:
        st.metric("Year 1 Interest", f"${first_year_interest:,.2f}")
    
    with tax_col2:
        st.metric("Year 1 Tax Savings", f"${first_year_tax_savings:,.2f}")
    
    with tax_col3:
        st.metric(f"{analysis_years}-Year Tax Savings", f"${cumulative_tax_savings:,.2f}")
    
    # Break-even
    st.divider()
    st.subheader("⚖️ Break-Even Analysis")
    
    if break_even_year:
        st.success(f"🎯 **Break-even: Year {break_even_year}**")
    else:
        st.warning(f"⚠️ **No break-even within {analysis_years} years**")
    
    # Charts - Mobile optimized
    st.divider()
    st.subheader(f"📈 {analysis_years}-Year Analysis")
    
    # Create responsive charts
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cumulative Costs & Equity", "Inflation-Adjusted Comparison"),
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45]
    )
    
    # Top chart
    fig.add_trace(
        go.Scatter(x=years_list, y=ownership_cumulative, name='Ownership Cost',
                  line=dict(color='#FF6B6B', width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=years_list, y=rental_cumulative, name='Rental Cost',
                  line=dict(color='#4ECDC4', width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=years_list, y=home_equity, name='Home Equity',
                  line=dict(color='#95E1D3', width=2, dash='dash'),
                  fill='tozeroy'),
        row=1, col=1
    )
    
    # Bottom chart - inflation adjusted
    fig.add_trace(
        go.Scatter(x=years_list, y=ownership_real_value, name='Real Ownership Cost',
                  line=dict(color='#FF6B6B', width=2, dash='dot')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=years_list, y=rental_real_value, name='Real Rental Cost',
                  line=dict(color='#4ECDC4', width=2, dash='dot')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Present Value ($)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    # Selling costs section
    if plan_to_sell and selling_costs:
        st.divider()
        st.subheader("💵 Selling Costs & Net Proceeds")
        
        sell_col1, sell_col2, sell_col3 = st.columns(3)
        
        with sell_col1:
            st.metric("Home Value", f"${current_home_value:,.2f}")
            st.write(f"**Agent Commission (6%):** ${selling_costs['agent_commission']:,.2f}")
            st.write(f"**Title Fees:** ${selling_costs['title_fees']:,.2f}")
            st.write(f"**Transfer Taxes:** ${selling_costs['transfer_taxes']:,.2f}")
            st.write(f"**Other Fees:** ${selling_costs['other_fees']:,.2f}")
        
        with sell_col2:
            st.metric("Capital Gain", f"${selling_costs['capital_gain']:,.2f}")
            st.metric("Taxable Gain", f"${selling_costs['taxable_gain']:,.2f}",
                     help="After $500k exclusion (married)")
            st.metric("Capital Gains Tax", f"${selling_costs['capital_gains_tax']:,.2f}")
        
        with sell_col3:
            st.metric("Total Selling Costs", f"${selling_costs['total_selling_costs']:,.2f}")
            st.metric("Net Proceeds", f"${selling_costs['net_proceeds']:,.2f}",
                     help="After all costs and taxes")
            
            # Adjust final comparison
            net_equity = selling_costs['net_proceeds'] - remaining_principal
            st.metric("True Net Equity", f"${net_equity:,.2f}",
                     help="Proceeds minus remaining loan")
    
    # Final summary
    st.divider()
    st.subheader("🎯 Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.metric(f"Total Ownership ({analysis_years} yrs)", f"${ownership_cumulative[-1]:,.2f}")
        
        if plan_to_sell and selling_costs:
            final_equity = selling_costs['net_proceeds'] - remaining_principal
            st.metric("Net Proceeds After Sale", f"${final_equity:,.2f}")
            net_cost = ownership_cumulative[-1] - final_equity
        else:
            st.metric("Home Equity Built", f"${home_equity[-1]:,.2f}")
            net_cost = ownership_cumulative[-1] - home_equity[-1]
        
        st.metric("Net Cost", f"${net_cost:,.2f}")
        st.metric("Tax Savings", f"${cumulative_tax_savings:,.2f}")
    
    with summary_col2:
        st.metric(f"Total Rental ({analysis_years} yrs)", f"${rental_cumulative[-1]:,.2f}")
        
        savings = rental_cumulative[-1] - net_cost
        
        if savings > 0:
            st.success(f"✅ **Buying saves ${savings:,.2f}**")
        else:
            st.warning(f"⚠️ **Renting saves ${abs(savings):,.2f}**")
        
        # Real values (inflation-adjusted)
        st.metric("Real Ownership Cost", f"${ownership_real_value[-1]:,.2f}",
                 help="Adjusted for inflation")
        st.metric("Real Rental Cost", f"${rental_real_value[-1]:,.2f}",
                 help="Adjusted for inflation")

# ==================== TAB 2: AMORTIZATION ====================

with tab2:
    st.subheader("🔢 Amortization Schedule")
    
    if loan_type == "Fixed Rate":
        st.write(f"**Loan Type:** Fixed Rate")
        st.write(f"**Interest Rate:** {interest_rate}%")
    else:
        st.write(f"**Loan Type:** {fixed_period}-Year ARM")
        st.write(f"**Initial Rate:** {initial_rate}%")
        st.write(f"**Rate Adjusts:** Every {adjustment_frequency} year(s) after year {fixed_period}")
        st.write(f"**Periodic Cap:** ±{periodic_cap}%")
        st.write(f"**Lifetime Cap:** {lifetime_cap}%")
    
    st.write(f"**Loan Amount:** ${loan_amount:,.2f}")
    st.write(f"**Initial Monthly Payment:** ${monthly_mortgage:,.2f}")
    
    if monthly_pmi > 0:
        st.write(f"**Monthly PMI:** ${monthly_pmi:,.2f} (until 78% LTV)")
    
    st.divider()
    
    # Yearly summary
    st.subheader("📅 Yearly Summary")
    
    if loan_type == "Fixed Rate":
        yearly_summary = amortization.groupby('Year').agg({
            'Payment': 'sum',
            'Principal': 'sum',
            'Interest': 'sum',
            'Balance': 'last'
        }).reset_index()
        yearly_summary.columns = ['Year', 'Total Payments', 'Principal Paid', 'Interest Paid', 'Ending Balance']
    else:
        yearly_summary = amortization.copy()
        yearly_summary['Total Payments'] = yearly_summary['Monthly_Payment'] * 12
        yearly_summary = yearly_summary[['Year', 'Rate', 'Total Payments', 'Principal', 'Interest', 'Balance']]
        yearly_summary.columns = ['Year', 'Rate (%)', 'Total Payments', 'Principal Paid', 'Interest Paid', 'Ending Balance']
    
    st.dataframe(yearly_summary.head(20), use_container_width=True, hide_index=True)
    
    if len(yearly_summary) > 20:
        st.caption(f"Showing first 20 years. Download full schedule in Export tab.")
    
    # Chart - mobile friendly
    st.divider()
    st.subheader("📊 Payment Breakdown")
    
    if loan_type == "Fixed Rate":
        yearly_agg = amortization.groupby('Year').agg({
            'Principal': 'sum',
            'Interest': 'sum'
        }).reset_index()
    else:
        yearly_agg = amortization[['Year', 'Principal', 'Interest']].copy()
    
    fig_amort = go.Figure()
    
    fig_amort.add_trace(go.Bar(
        x=yearly_agg['Year'],
        y=yearly_agg['Principal'],
        name='Principal',
        marker_color='#95E1D3'
    ))
    
    fig_amort.add_trace(go.Bar(
        x=yearly_agg['Year'],
        y=yearly_agg['Interest'],
        name='Interest',
        marker_color='#FF6B6B'
    ))
    
    fig_amort.update_layout(
        barmode='stack',
        xaxis_title="Year",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig_amort, use_container_width=True, config={'responsive': True})
    
    # ARM rate changes visualization
    if loan_type == "ARM":
        st.divider()
        st.subheader("📈 ARM Rate Changes")
        
        fig_rate = go.Figure()
        
        fig_rate.add_trace(go.Scatter(
            x=amortization['Year'],
            y=amortization['Rate'],
            mode='lines+markers',
            name='Interest Rate',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8)
        ))
        
        fig_rate.update_layout(
            xaxis_title="Year",
            yaxis_title="Interest Rate (%)",
            height=350,
            hovermode='x unified',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_rate, use_container_width=True, config={'responsive': True})

# ==================== TAB 3: REFINANCE CALCULATOR ====================

with tab3:
    st.subheader("🔄 Refinance Calculator")
    
    st.info("💡 Use this to see if refinancing your current mortgage makes sense")
    
    refi_col1, refi_col2 = st.columns(2)
    
    with refi_col1:
        st.markdown("#### Current Loan")
        
        current_balance = st.number_input(
            "Current Loan Balance ($)",
            min_value=10000,
            max_value=5000000,
            value=int(loan_amount * 0.8),
            step=10000,
            help="How much you still owe"
        )
        
        current_rate = st.number_input(
            "Current Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=7.0,
            step=0.1
        )
        
        years_left = st.number_input(
            "Years Remaining",
            min_value=1,
            max_value=30,
            value=25,
            step=1
        )
    
    with refi_col2:
        st.markdown("#### New Loan")
        
        new_rate = st.number_input(
            "New Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.5,
            step=0.1
        )
        
        new_term = st.number_input(
            "New Loan Term (years)",
            min_value=10,
            max_value=30,
            value=30,
            step=5
        )
        
        refi_costs = st.number_input(
            "Refinancing Costs ($)",
            min_value=0,
            max_value=50000,
            value=5000,
            step=500,
            help="Closing costs, appraisal, etc."
        )
    
    # Calculate
    refi_analysis = calculate_refinance_savings(
        current_balance,
        current_rate,
        years_left,
        new_rate,
        new_term,
        refi_costs
    )
    
    st.divider()
    
    # Results
    refi_result_col1, refi_result_col2, refi_result_col3 = st.columns(3)
    
    with refi_result_col1:
        st.metric("Current Monthly Payment", f"${refi_analysis['current_payment']:,.2f}")
        st.metric("New Monthly Payment", f"${refi_analysis['new_payment']:,.2f}")
    
    with refi_result_col2:
        st.metric("Monthly Savings", f"${refi_analysis['monthly_savings']:,.2f}")
        
        if refi_analysis['break_even_months'] < float('inf'):
            years = int(refi_analysis['break_even_months'] // 12)
            months = int(refi_analysis['break_even_months'] % 12)
            st.metric("Break-Even", f"{years}y {months}m",
                     help="Time to recover refinancing costs")
        else:
            st.metric("Break-Even", "N/A",
                     help="New payment is higher")
    
    with refi_result_col3:
        lifetime_savings = refi_analysis['lifetime_savings']
        
        if lifetime_savings > 0:
            st.metric("Total Savings", f"${lifetime_savings:,.2f}",
                     help=f"Over {min(years_left, new_term)} years")
            st.success("✅ Refinancing saves money!")
        else:
            st.metric("Total Cost", f"${abs(lifetime_savings):,.2f}",
                     help="Refinancing costs more")
            st.warning("⚠️ Refinancing costs more")
    
    # Recommendation
    st.divider()
    st.subheader("💡 Recommendation")
    
    if lifetime_savings > 0 and refi_analysis['break_even_months'] < 36:
        st.success(f"""
        **✅ Refinancing looks good!**
        - You'll save ${refi_analysis['monthly_savings']:,.2f}/month
        - Break-even in {int(refi_analysis['break_even_months'])} months
        - Total savings: ${lifetime_savings:,.2f}
        """)
    elif lifetime_savings > 0 and refi_analysis['break_even_months'] < 60:
        st.info(f"""
        **💭 Refinancing might make sense**
        - Monthly savings: ${refi_analysis['monthly_savings']:,.2f}
        - Break-even: {int(refi_analysis['break_even_months'])} months (longer)
        - Consider if you plan to stay {int(refi_analysis['break_even_months'] // 12)}+ years
        """)
    else:
        st.warning("""
        **⚠️ Refinancing may not be worth it**
        - Break-even period is too long, or
        - New payment is higher, or
        - Total savings don't justify the hassle
        """)

# ==================== TAB 4: MONTE CARLO ====================

with tab4:
    st.subheader("🎲 Monte Carlo Simulation")
    
    st.markdown("""
    This simulation runs 1,000 scenarios with varying home appreciation rates and rent increases 
    to show the range of possible outcomes.
    """)
    
    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner("Running 1,000 simulations..."):
            # Prepare parameters
            base_params = {
                'home_price': home_price,
                'down_payment': down_payment,
                'home_appreciation': home_appreciation,
                'rent_increase': rent_increase_annual,
                'analysis_years': analysis_years,
                'annual_ownership_cost': sum(ownership_costs_yearly) / len(ownership_costs_yearly),
                'remaining_loan': remaining_principal
            }
            
            # Run simulation
            mc_results = monte_carlo_simulation(base_params, n_simulations=1000)
            
            # Display results
            st.success("✅ Simulation complete!")
            
            st.divider()
            
            # Summary statistics
            st.subheader("📊 Simulation Results")
            
            mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
            
            with mc_col1:
                median_value = mc_results['final_home_value'].median()
                st.metric("Median Home Value", f"${median_value:,.0f}")
            
            with mc_col2:
                median_equity = mc_results['total_equity'].median()
                st.metric("Median Equity", f"${median_equity:,.0f}")
            
            with mc_col3:
                best_case = mc_results['total_equity'].quantile(0.90)
                st.metric("90th Percentile Equity", f"${best_case:,.0f}",
                         help="Top 10% of scenarios")
            
            with mc_col4:
                worst_case = mc_results['total_equity'].quantile(0.10)
                st.metric("10th Percentile Equity", f"${worst_case:,.0f}",
                         help="Bottom 10% of scenarios")
            
            st.divider()
            
            # Distribution charts
            st.subheader("📈 Distribution of Outcomes")
            
            fig_mc = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Home Value Distribution", "Total Equity Distribution")
            )
            
            # Home value histogram
            fig_mc.add_trace(
                go.Histogram(
                    x=mc_results['final_home_value'],
                    nbinsx=50,
                    name='Home Value',
                    marker_color='#4ECDC4'
                ),
                row=1, col=1
            )
            
            # Equity histogram
            fig_mc.add_trace(
                go.Histogram(
                    x=mc_results['total_equity'],
                    nbinsx=50,
                    name='Equity',
                    marker_color='#95E1D3'
                ),
                row=1, col=2
            )
            
            fig_mc.update_xaxes(title_text="Home Value ($)", row=1, col=1)
            fig_mc.update_xaxes(title_text="Equity ($)", row=1, col=2)
            fig_mc.update_yaxes(title_text="Frequency", row=1, col=1)
            fig_mc.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig_mc.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            st.plotly_chart(fig_mc, use_container_width=True, config={'responsive': True})
            
            # Appreciation rate impact
            st.divider()
            st.subheader("🎯 Appreciation Rate Impact")
            
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=mc_results['appreciation_rate'],
                y=mc_results['total_equity'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=mc_results['total_equity'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Equity ($)")
                ),
                text=[f"Rate: {r:.1f}%<br>Equity: ${e:,.0f}" 
                      for r, e in zip(mc_results['appreciation_rate'], mc_results['total_equity'])],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig_scatter.update_layout(
                xaxis_title="Home Appreciation Rate (%)",
                yaxis_title="Total Equity ($)",
                height=450,
                hovermode='closest',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True, config={'responsive': True})
            
            # Key insights
            st.divider()
            st.subheader("💡 Key Insights")
            
            positive_equity = (mc_results['total_equity'] > 0).sum()
            percent_positive = (positive_equity / len(mc_results)) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Scenarios with Positive Equity", f"{percent_positive:.1f}%")
                
                avg_appreciation = mc_results['appreciation_rate'].mean()
                st.write(f"**Average Appreciation Rate:** {avg_appreciation:.2f}%")
                
                median_net = mc_results['net_cost'].median()
                st.write(f"**Median Net Cost:** ${median_net:,.0f}")
            
            with col2:
                if percent_positive > 75:
                    st.success("✅ High probability of positive equity")
                elif percent_positive > 50:
                    st.info("💭 Moderate probability of positive equity")
                else:
                    st.warning("⚠️ Lower probability of positive equity")
                
                # Risk assessment
                std_equity = mc_results['total_equity'].std()
                st.write(f"**Equity Std Deviation:** ${std_equity:,.0f}")
                st.caption("Higher = more uncertainty")
            
            # Percentile table
            st.divider()
            st.subheader("📋 Outcome Percentiles")
            
            percentiles = [10, 25, 50, 75, 90]
            percentile_data = []
            
            for p in percentiles:
                percentile_data.append({
                    'Percentile': f"{p}th",
                    'Home Value': f"${mc_results['final_home_value'].quantile(p/100):,.0f}",
                    'Total Equity': f"${mc_results['total_equity'].quantile(p/100):,.0f}",
                    'Net Cost': f"${mc_results['net_cost'].quantile(p/100):,.0f}"
                })
            
            percentile_df = pd.DataFrame(percentile_data)
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)
            
            st.caption("10th percentile = worst 10% of scenarios, 90th percentile = best 10% of scenarios")
            
            # Store results in session state for export
            st.session_state.mc_results = mc_results

# ==================== TAB 5: COMPARE SCENARIOS ====================

with tab5:
    st.subheader("⚖️ Compare Scenarios")
    
    if 'scenarios' not in st.session_state or len(st.session_state.scenarios) == 0:
        st.info("👈 Save scenarios from the sidebar to compare them here!")
        
        st.markdown("""
        **How to use:**
        1. Configure your first scenario in the sidebar
        2. Click '💾 Save This Scenario'
        3. Change inputs for a different scenario
        4. Save that scenario too
        5. Compare them all here!
        
        **Example scenarios to compare:**
        - Different down payment amounts (10% vs 20%)
        - Fixed rate vs ARM
        - Different home prices
        - Various loan terms (15-year vs 30-year)
        - Different locations (urban vs suburban rent costs)
        """)
    else:
        st.success(f"✅ You have {len(st.session_state.scenarios)} saved scenario(s)")
        
        # Comparison table
        comparison_data = []
        
        for scenario in st.session_state.scenarios:
            s_down_payment = scenario['home_price'] * (scenario['down_payment_pct'] / 100)
            s_loan_amount = scenario['home_price'] - s_down_payment
            
            # Calculate based on loan type
            if scenario.get('loan_type') == "ARM":
                s_rate = scenario.get('initial_rate', 6.0)
            else:
                s_rate = scenario.get('interest_rate', 6.5)
            
            s_term = scenario.get('loan_term', 30)
            s_monthly_payment = calculate_monthly_mortgage(s_loan_amount, s_rate, s_term)
            
            # PMI
            s_pmi = calculate_pmi_monthly(s_loan_amount, scenario['home_price'])
            
            comparison_data.append({
                'Scenario': scenario['name'],
                'Loan Type': scenario.get('loan_type', 'Fixed Rate'),
                'Home Price': f"${scenario['home_price']:,.0f}",
                'Down Payment': f"{scenario['down_payment_pct']}% (${s_down_payment:,.0f})",
                'Interest Rate': f"{s_rate}%",
                'Loan Term': f"{s_term} yrs",
                'Monthly Payment': f"${s_monthly_payment:,.2f}",
                'PMI': f"${s_pmi:,.2f}" if s_pmi > 0 else "None",
                'Monthly Rent': f"${scenario['monthly_rent']:,.2f}",
                'Analysis Period': f"{scenario['analysis_years']} yrs"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.divider()
        st.subheader("📊 Visual Comparison")
        
        # Extract numeric values for charts
        monthly_payments = [float(d['Monthly Payment'].replace('$', '').replace(',', '')) 
                          for d in comparison_data]
        monthly_rents = [float(d['Monthly Rent'].replace('$', '').replace(',', '')) 
                        for d in comparison_data]
        scenario_names = [d['Scenario'] for d in comparison_data]
        
        # Create comparison chart
        fig_compare = go.Figure()
        
        fig_compare.add_trace(go.Bar(
            name='Monthly Payment (P&I)',
            x=scenario_names,
            y=monthly_payments,
            marker_color='#FF6B6B',
            text=[f"${p:,.0f}" for p in monthly_payments],
            textposition='outside'
        ))
        
        fig_compare.add_trace(go.Bar(
            name='Monthly Rent',
            x=scenario_names,
            y=monthly_rents,
            marker_color='#4ECDC4',
            text=[f"${r:,.0f}" for r in monthly_rents],
            textposition='outside'
        ))
        
        fig_compare.update_layout(
            barmode='group',
            xaxis_title="Scenario",
            yaxis_title="Monthly Cost ($)",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig_compare, use_container_width=True, config={'responsive': True})
        
        # Down payment comparison
        st.divider()
        st.subheader("💰 Down Payment Comparison")
        
        down_payments = []
        for scenario in st.session_state.scenarios:
            dp = scenario['home_price'] * (scenario['down_payment_pct'] / 100)
            down_payments.append(dp)
        
        fig_dp = go.Figure(go.Bar(
            x=scenario_names,
            y=down_payments,
            marker_color='#95E1D3',
            text=[f"${dp:,.0f}" for dp in down_payments],
            textposition='outside'
        ))
        
        fig_dp.update_layout(
            xaxis_title="Scenario",
            yaxis_title="Down Payment ($)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_dp, use_container_width=True, config={'responsive': True})
        
        # Management buttons
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Scenarios", use_container_width=True):
                st.session_state.scenarios = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Current", use_container_width=True):
                st.session_state.scenarios = [{
                    'name': scenario_name,
                    'loan_type': loan_type,
                    'home_price': home_price,
                    'down_payment_pct': down_payment_pct,
                    'interest_rate': interest_rate if loan_type == "Fixed Rate" else None,
                    'initial_rate': initial_rate if loan_type == "ARM" else None,
                    'loan_term': loan_term,
                    'monthly_rent': monthly_rent,
                    'analysis_years': analysis_years
                }]
                st.success("Reset to current configuration!")
                st.rerun()

# ==================== TAB 6: EXPORT ====================

with tab6:
    st.subheader("💾 Export Your Results")
    
    st.markdown("""
    Download your complete analysis in various formats. All exports include:
    - Input parameters
    - Monthly cost breakdown
    - Year-by-year analysis
    - Tax savings calculations
    - Amortization schedule
    - Final summary
    """)
    
    # Prepare comprehensive export data
    
    # 1. Summary sheet
    export_summary = {
        'Parameter': [
            'Scenario Name',
            'Loan Type',
            'Home Price',
            'Down Payment',
            'Down Payment %',
            'Loan Amount',
            'Interest Rate' if loan_type == "Fixed Rate" else "Initial Rate",
            'Loan Term',
            'Monthly Mortgage Payment',
            'Monthly PMI',
            'Monthly Property Tax',
            'Monthly Insurance',
            'Monthly HOA',
            'Monthly Maintenance',
            'Total Monthly Ownership',
            'Monthly Rent',
            'Total Monthly Rental',
            'Closing Costs',
            'Analysis Period',
            f'Total Ownership Costs ({analysis_years} years)',
            f'Total Rental Costs ({analysis_years} years)',
            f'Home Equity Built ({analysis_years} years)',
            'Net Cost of Ownership',
            f'Total Tax Savings ({analysis_years} years)',
            'Break-Even Year',
            'Inflation Rate Used',
            'Expected Home Appreciation',
            'Expected Rent Increase'
        ],
        'Value': [
            scenario_name,
            loan_type,
            f"${home_price:,.2f}",
            f"${down_payment:,.2f}",
            f"{down_payment_pct}%",
            f"${loan_amount:,.2f}",
            f"{interest_rate if loan_type == 'Fixed Rate' else initial_rate}%",
            f"{loan_term} years",
            f"${monthly_mortgage:,.2f}",
            f"${monthly_pmi:,.2f}",
            f"${monthly_property_tax:,.2f}",
            f"${monthly_insurance:,.2f}",
            f"${hoa_monthly:,.2f}",
            f"${monthly_maintenance:,.2f}",
            f"${total_monthly_ownership:,.2f}",
            f"${monthly_rent:,.2f}",
            f"${total_monthly_rental:,.2f}",
            f"${closing_costs['total']:,.2f}" if closing_costs else "$0.00",
            f"{analysis_years} years",
            f"${ownership_cumulative[-1]:,.2f}",
            f"${rental_cumulative[-1]:,.2f}",
            f"${home_equity[-1]:,.2f}",
            f"${ownership_cumulative[-1] - home_equity[-1]:,.2f}",
            f"${cumulative_tax_savings:,.2f}",
            f"Year {break_even_year}" if break_even_year else "No break-even",
            f"{inflation_rate}%",
            f"{home_appreciation}%",
            f"{rent_increase_annual}%"
        ]
    }
    
    summary_df = pd.DataFrame(export_summary)
    
    # 2. Year-by-year analysis
    yearly_data = pd.DataFrame({
        'Year': years_list,
        'Cumulative Ownership Cost': ownership_cumulative,
        'Cumulative Rental Cost': rental_cumulative,
        'Home Equity': home_equity,
        'Cumulative Tax Savings': tax_savings_cumulative,
        'Annual Ownership Cost': ownership_costs_yearly,
        'Annual Rental Cost': rental_costs_yearly,
        'Real Ownership Cost (Inflation Adj)': ownership_real_value,
        'Real Rental Cost (Inflation Adj)': rental_real_value
    })
    
    # 3. Amortization schedule (first 360 months for 30-year)
    amort_export = amortization.copy()
    if loan_type == "Fixed Rate":
        amort_export = amort_export[['Month', 'Year', 'Payment', 'Principal', 'Interest', 'Balance']]
    else:
        amort_export = amort_export[['Year', 'Rate', 'Monthly_Payment', 'Principal', 'Interest', 'Balance']]
        amort_export.columns = ['Year', 'Interest Rate (%)', 'Monthly Payment', 'Principal', 'Interest', 'Balance']
    
    # 4. Tax analysis
    tax_details = []
    for year in range(1, min(analysis_years + 1, 31)):
        if loan_type == "Fixed Rate":
            year_amort = amortization[amortization['Year'] == year]
            if not year_amort.empty:
                interest = year_amort['Interest'].sum()
            else:
                interest = 0
        else:
            year_amort = amortization[amortization['Year'] == year]
            if not year_amort.empty:
                interest = year_amort['Interest'].iloc[0]
            else:
                interest = 0
        
        savings = calculate_tax_savings(interest, property_tax_annual, annual_income, filing_status)
        
        tax_details.append({
            'Year': year,
            'Mortgage Interest': f"${interest:,.2f}",
            'Property Tax': f"${property_tax_annual:,.2f}",
            'Total Deductions': f"${interest + property_tax_annual:,.2f}",
            'Tax Savings': f"${savings:,.2f}"
        })
    
    tax_df = pd.DataFrame(tax_details)
    
    # 5. Selling costs if applicable
    if plan_to_sell and selling_costs:
        selling_df = pd.DataFrame({
            'Item': [
                'Final Home Value',
                'Original Purchase Price',
                'Capital Gain',
                'Capital Gains Exclusion',
                'Taxable Gain',
                'Capital Gains Tax',
                'Agent Commission (6%)',
                'Title Fees',
                'Transfer Taxes',
                'Other Fees',
                'Total Selling Costs',
                'Remaining Loan Balance',
                'Net Proceeds'
            ],
            'Amount': [
                f"${current_home_value:,.2f}",
                f"${home_price:,.2f}",
                f"${selling_costs['capital_gain']:,.2f}",
                "$500,000",
                f"${selling_costs['taxable_gain']:,.2f}",
                f"${selling_costs['capital_gains_tax']:,.2f}",
                f"${selling_costs['agent_commission']:,.2f}",
                f"${selling_costs['title_fees']:,.2f}",
                f"${selling_costs['transfer_taxes']:,.2f}",
                f"${selling_costs['other_fees']:,.2f}",
                f"${selling_costs['total_selling_costs']:,.2f}",
                f"${remaining_principal:,.2f}",
                f"${selling_costs['net_proceeds']:,.2f}"
            ]
        })
    
    # 6. Closing costs breakdown
    if closing_costs:
        closing_df = pd.DataFrame({
            'Cost Item': [
                'Origination Fee (1%)',
                'Appraisal',
                'Title Insurance',
                'Home Inspection',
                'Recording Fees',
                'Other Fees',
                'Total Closing Costs'
            ],
            'Amount': [
                f"${closing_costs['origination_fee']:,.2f}",
                f"${closing_costs['appraisal']:,.2f}",
                f"${closing_costs['title_insurance']:,.2f}",
                f"${closing_costs['inspection']:,.2f}",
                f"${closing_costs['recording_fees']:,.2f}",
                f"${closing_costs['other_fees']:,.2f}",
                f"${closing_costs['total']:,.2f}"
            ]
        })
    
    st.divider()
    
    # Export buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Excel Workbook")
        
        # Create Excel file with multiple sheets
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Write all sheets
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            yearly_data.to_excel(writer, sheet_name='Yearly Analysis', index=False)
            amort_export.to_excel(writer, sheet_name='Amortization Schedule', index=False)
            tax_df.to_excel(writer, sheet_name='Tax Benefits', index=False)
            
            if closing_costs:
                closing_df.to_excel(writer, sheet_name='Closing Costs', index=False)
            
            if plan_to_sell and selling_costs:
                selling_df.to_excel(writer, sheet_name='Selling Costs', index=False)
            
            # Add Monte Carlo results if they exist
            if 'mc_results' in st.session_state:
                st.session_state.mc_results.to_excel(writer, sheet_name='Monte Carlo Results', index=False)
            
            # Format workbook
            workbook = writer.book
            
            # Add formats
            money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            percent_fmt = workbook.add_format({'num_format': '0.00%'})
            header_fmt = workbook.add_format({
                'bold': True,
                'bg_color': '#4ECDC4',
                'font_color': 'white'
            })
            
            # Apply formatting to each sheet
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:Z', 20)  # Set column width
        
        excel_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Complete Excel Workbook",
            data=excel_buffer,
            file_name=f"home_loan_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        
        st.caption(f"""
        **Includes {6 + (1 if closing_costs else 0) + (1 if plan_to_sell and selling_costs else 0) + (1 if 'mc_results' in st.session_state else 0)} sheets:**
        - Summary
        - Yearly Analysis
        - Amortization Schedule
        - Tax Benefits
        {'- Closing Costs' if closing_costs else ''}
        {'- Selling Costs' if plan_to_sell and selling_costs else ''}
        {'- Monte Carlo Results' if 'mc_results' in st.session_state else ''}
        """)
    
    with col2:
        st.subheader("📄 CSV Bundle")
        
        # Create comprehensive CSV
        csv_buffer = io.StringIO()
        
        csv_buffer.write(f"HOME LOAN ANALYSIS REPORT\n")
        csv_buffer.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        csv_buffer.write(f"{'='*80}\n\n")
        
        csv_buffer.write("SUMMARY\n")
        csv_buffer.write("="*80 + "\n")
        summary_df.to_csv(csv_buffer, index=False)
        
        csv_buffer.write("\n\nYEARLY ANALYSIS\n")
        csv_buffer.write("="*80 + "\n")
        yearly_data.to_csv(csv_buffer, index=False)
        
        csv_buffer.write("\n\nTAX BENEFITS\n")
        csv_buffer.write("="*80 + "\n")
        tax_df.to_csv(csv_buffer, index=False)
        
        if closing_costs:
            csv_buffer.write("\n\nCLOSING COSTS\n")
            csv_buffer.write("="*80 + "\n")
            closing_df.to_csv(csv_buffer, index=False)
        
        if plan_to_sell and selling_costs:
            csv_buffer.write("\n\nSELLING COSTS\n")
            csv_buffer.write("="*80 + "\n")
            selling_df.to_csv(csv_buffer, index=False)
        
        csv_buffer.write("\n\nAMORTIZATION SCHEDULE (First 5 years)\n")
        csv_buffer.write("="*80 + "\n")
        if loan_type == "Fixed Rate":
            first_years = amortization[amortization['Year'] <= 5]
        else:
            first_years = amortization[amortization['Year'] <= 5]
        first_years.to_csv(csv_buffer, index=False)
        csv_buffer.write("\n(Full schedule available in Excel download)\n")
        
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"home_loan_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.caption("Comprehensive CSV with all analysis data")
    
    # Quick Summary PDF-style text export
    st.divider()
    st.subheader("📋 Quick Summary (Copy-Paste)")
    
    summary_text = f"""
HOME LOAN ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
{'='*80}

SCENARIO: {scenario_name}
Loan Type: {loan_type}

PURCHASE DETAILS
- Home Price: ${home_price:,.2f}
- Down Payment: ${down_payment:,.2f} ({down_payment_pct}%)
- Loan Amount: ${loan_amount:,.2f}
- Interest Rate: {interest_rate if loan_type == 'Fixed Rate' else initial_rate}%
- Loan Term: {loan_term} years
{f'- Closing Costs: ${closing_costs["total"]:,.2f}' if closing_costs else ''}

MONTHLY COSTS
Ownership:
- Mortgage (P&I): ${monthly_mortgage:,.2f}
{f'- PMI: ${monthly_pmi:,.2f}' if monthly_pmi > 0 else ''}
- Property Tax: ${monthly_property_tax:,.2f}
- Insurance: ${monthly_insurance:,.2f}
- HOA: ${hoa_monthly:,.2f}
- Maintenance: ${monthly_maintenance:,.2f}
- TOTAL: ${total_monthly_ownership:,.2f}

Rental:
- Rent: ${monthly_rent:,.2f}
- Insurance: ${monthly_renters_insurance:,.2f}
- TOTAL: ${total_monthly_rental:,.2f}

{analysis_years}-YEAR ANALYSIS
- Total Ownership Cost: ${ownership_cumulative[-1]:,.2f}
- Total Rental Cost: ${rental_cumulative[-1]:,.2f}
- Home Equity Built: ${home_equity[-1]:,.2f}
- Net Cost of Ownership: ${ownership_cumulative[-1] - home_equity[-1]:,.2f}
- Total Tax Savings: ${cumulative_tax_savings:,.2f}

{'Break-Even: Year ' + str(break_even_year) if break_even_year else 'No break-even within analysis period'}

RECOMMENDATION
{f'✅ Buying saves ${rental_cumulative[-1] - (ownership_cumulative[-1] - home_equity[-1]):,.2f} over {analysis_years} years' 
 if rental_cumulative[-1] > (ownership_cumulative[-1] - home_equity[-1]) 
 else f'⚠️ Renting saves ${abs(rental_cumulative[-1] - (ownership_cumulative[-1] - home_equity[-1])):,.2f} over {analysis_years} years'}

ASSUMPTIONS
- Home Appreciation: {home_appreciation}% per year
- Rent Increase: {rent_increase_annual}% per year
- Inflation Rate: {inflation_rate}%
- Tax Filing Status: {filing_status}
- Annual Income: ${annual_income:,.2f}

{f"""
SELLING ANALYSIS (Year {analysis_years})
- Final Home Value: ${current_home_value:,.2f}
- Total Selling Costs: ${selling_costs['total_selling_costs']:,.2f}
- Net Proceeds: ${selling_costs['net_proceeds']:,.2f}
- Capital Gains Tax: ${selling_costs['capital_gains_tax']:,.2f}
""" if plan_to_sell and selling_costs else ''}

DISCLAIMER
This analysis is for educational purposes only. Consult with a financial 
advisor, tax professional, and real estate expert before making any major 
financial decisions. Individual circumstances vary significantly.

{'='*80}
Generated by Home Loan Calculator Pro
    """
    
    st.text_area(
        "Copy this summary:",
        summary_text,
        height=400,
        help="Select all (Ctrl+A) and copy (Ctrl+C)"
    )
    
    # Data preview
    st.divider()
    st.subheader("👀 Data Preview")
    
    preview_tab1, preview_tab2, preview_tab3, preview_tab4 = st.tabs([
        "Summary", 
        "Yearly", 
        "Tax Benefits",
        "Amortization"
    ])
    
    with preview_tab1:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with preview_tab2:
        st.dataframe(yearly_data, use_container_width=True, hide_index=True)
    
    with preview_tab3:
        st.dataframe(tax_df, use_container_width=True, hide_index=True)
    
    with preview_tab4:
        display_amort = amort_export.head(60)  # Show first 5 years
        st.dataframe(display_amort, use_container_width=True, hide_index=True)
        st.caption("Showing first 60 months. Full schedule in Excel download.")

# ==================== FOOTER ====================

st.divider()

# Add helpful information
with st.expander("ℹ️ About This Calculator"):
    st.markdown("""
    ### Features
    
    **Loan Types:**
    - Fixed-rate mortgages
    - ARM (Adjustable Rate Mortgages) with caps and adjustments
    
    **Cost Analysis:**
    - PMI (Private Mortgage Insurance) calculations
    - Closing costs estimation
    - Property tax and insurance
    - HOA fees and maintenance
    - Tax benefit calculations
    
    **Advanced Tools:**
    - Refinance calculator
    - Monte Carlo simulation (1,000 scenarios)
    - Inflation adjustments
    - Selling cost projections
    - Multi-scenario comparison
    
    **Export Options:**
    - Excel workbook with multiple sheets
    - CSV reports
    - Copy-paste text summaries
    
    ### Methodology
    
    **Tax Savings:** Uses 2024 standard deductions and tax brackets. Compares itemized 
    deductions (mortgage interest + property tax) against standard deduction.
    
    **PMI Rates:** Varies by LTV ratio. Automatically drops off at 78% LTV.
    
    **Inflation Adjustment:** Uses compound discount formula to convert future dollars 
    to present value.
    
    **Monte Carlo:** Simulates 1,000 scenarios with normally distributed appreciation 
    rates (±2% standard deviation) and rent increases (±1.5% standard deviation).
    
    **ARM Calculations:** Models rate adjustments based on fixed period, adjustment 
    frequency, and caps (periodic and lifetime).
    
    ### Limitations
    
    - Simplified tax calculations (doesn't account for AMT, state taxes, etc.)
    - Assumes consistent maintenance costs
    - Doesn't model market crashes or economic disruptions
    - Selling costs are estimates (actual costs vary by location)
    - Monte Carlo uses historical patterns (past ≠ future)
    
    ### Disclaimer
    
    This tool is for **educational and planning purposes only**. It does not constitute 
    financial, legal, or tax advice. Always consult with qualified professionals:
    
    - **Mortgage Broker** - for loan options and rates
    - **Financial Advisor** - for investment decisions
    - **Tax Professional** - for tax implications
    - **Real Estate Agent** - for local market insights
    - **Attorney** - for legal questions
    
    Individual circumstances vary greatly. Your actual costs, taxes, and returns 
    will differ.
    """)

with st.expander("📖 How to Use This Calculator"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Set Your Parameters** (Sidebar)
       - Choose loan type (Fixed or ARM)
       - Enter home price and down payment
       - Set interest rate and loan term
       - Add property tax, insurance, HOA
       - Enter your income for tax calculations
    
    2. **Review Main Analysis** (Tab 1)
       - See monthly cost breakdown
       - Check tax savings
       - Find break-even point
       - Review cumulative costs vs equity
    
    3. **Explore Amortization** (Tab 2)
       - See how much goes to principal vs interest
       - View yearly summaries
       - Check when PMI drops off (for ARM loans, see rate changes)
    
    4. **Consider Refinancing** (Tab 3)
       - Enter your current loan details
       - Compare with new rates
       - See break-even timeline
    
    5. **Run Simulations** (Tab 4)
       - Click "Run Simulation"
       - See range of possible outcomes
       - Understand uncertainty in projections
    
    6. **Compare Scenarios** (Tab 5)
       - Save multiple scenarios
       - Compare side-by-side
       - Find best option for your situation
    
    7. **Export Results** (Tab 6)
       - Download Excel workbook
       - Get CSV reports
       - Share with advisors
    
    ### Pro Tips
    
    - **PMI Impact:** Even 1-2% more down payment can eliminate PMI
    - **15 vs 30 Year:** Shorter term = higher monthly payment but less total interest
    - **ARM vs Fixed:** ARM is risky if you can't afford potential increases
    - **Tax Benefits:** May not apply if standard deduction is higher
    - **Break-Even:** If you might move before break-even, renting may be better
    - **Monte Carlo:** Run this to see how sensitive your decision is to market changes
    - **Refinancing:** Generally worth it if you can drop rate by 0.75% or more
    """)

with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues
    
    **"PMI is too high"**
    - Increase your down payment to 20%+ to eliminate PMI
    - Or plan for when LTV reaches 78% (usually 3-5 years with appreciation)
    
    **"No tax savings showing"**
    - Your standard deduction may exceed itemized deductions
    - Common with smaller mortgages or higher incomes
    - This is actually good - means standard deduction works for you!
    
    **"Break-even is too far away"**
    - Consider increasing down payment
    - Look at shorter loan terms (15-year vs 30-year)
    - Compare actual rent to mortgage payment
    - Factor in lifestyle benefits of ownership
    
    **"Monte Carlo shows high variability"**
    - Real estate is uncertain - this is normal
    - Consider more conservative assumptions
    - Focus on median outcomes, not extremes
    - Remember: historically, homes appreciate 3-4% annually
    
    **"ARM payments might get too high"**
    - Check the worst-case scenario with lifetime cap
    - Ensure you can afford maximum potential payment
    - Consider stress-testing at cap rate
    - Fixed rate might be safer if uncertain
    
    **"App is slow on mobile"**
    - Charts are optimized but may take a moment
    - Try reducing analysis period (10 years vs 30)
    - Monte Carlo simulations take 2-3 seconds (normal)
    
    ### Need Help?
    
    This is a sophisticated financial calculator. If you're unsure about:
    - **Mortgage options** → Talk to a mortgage broker
    - **Investment decisions** → Consult a financial advisor
    - **Tax implications** → See a CPA or tax professional
    - **Local market** → Work with a real estate agent
    """)

# Final footer
st.caption(f"""
**Professional Home Loan Calculator** | Built with Streamlit | 
Last updated: {datetime.now().strftime('%B %d, %Y')} | 
For educational purposes only - Not financial advice
""")

# Add version info
st.caption("Version 2.0 - Professional Edition with ARM, PMI, Refinancing, Monte Carlo & Inflation Adjustments")
    