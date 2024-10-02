# Lauren Jachimczyk
# CISC 5352 HW 1
from distutils.command.clean import clean

# 1a
# Annuity: PV= C/r - C/r * (1/(1+r)^T)
PV = 500000
r_1=0.05/12 #annual rate made monthly
r_2=0.07/12 #annual rate made monthly
T_1=15*12 #years made monthly
T_2=30*12 #years made monthly

def monthly_pmt(PV,r,T):
    C=(PV * r)/(1-(1+r)** -T)
    return C

# 15 year
monthly_pmt_15_yr = round(monthly_pmt(PV,r_1,T_1),2)
total_interest_paid_15_yr = round(((monthly_pmt_15_yr*T_1)-PV),2)
print(f'''The monthly payment for the 15 year mortgage is {monthly_pmt_15_yr}, 
and the total interest paid is {total_interest_paid_15_yr}''')

# 30 year
monthly_pmt_30_yr = round(monthly_pmt(PV,r_2,T_2),2)
total_interest_paid_30_yr = round(((monthly_pmt_30_yr*T_2)-PV),2)
print(f'''The monthly payment for the 30 year mortgage is {monthly_pmt_30_yr},
and the total interest paid is {total_interest_paid_30_yr}''')


# 1b
# EAR = (1+r/n)^n -1
n=12 #monthly compounding
apr_15 = 0.05
apr_30 = 0.07

def EAR(r,n):
    EAR=((1+r/n)** n)-1
    return EAR

# 15 year
EAR_15_yr = round(EAR(apr_15,n),5)
print(f'The EAR for the 15 year mortgage is {EAR_15_yr}')

# 30 year
EAR_30_yr = round(EAR(apr_30,n),5)
print(f'The EAR for the 30 year mortgage is {EAR_30_yr}')


# 1c

def convert_rate_from_EAR(EAR):
    return (1 + EAR)**(1 / 12) - 1

# 15 year
monthly_EAR_15_yr = convert_rate_from_EAR(EAR_15_yr)
monthly_pmt_15_yr_ear = round(monthly_pmt(PV,monthly_EAR_15_yr,T_1),2)
print(f'The monthly payment for the 15 year mortgage using the EAR is {monthly_pmt_15_yr_ear}')

# 30 year
monthly_EAR_30_yr = convert_rate_from_EAR(EAR_30_yr)
monthly_pmt_30_yr_ear = round(monthly_pmt(PV,monthly_EAR_30_yr,T_1),2)
print(f'The monthly payment for the 30 year mortgage using the EAR is {monthly_pmt_30_yr_ear}')


# 2a
# Valuation of ZCB: P0=F/(1+r)^T
# Price
F=1000 #par value
r_1 = 0.0525 #3yr spot rate
T_1 = 3 #3 years
def zcb_price(F,r,T):
    P0=F/(1+r)**T
    return P0

zcb_price_3_yr = round(zcb_price(F,r_1,T_1),2)
print(f'The price of the 3 year zero coupon bond is {zcb_price_3_yr}')

# YTM
#YTM=(C+((FV-PV)/t))/((FV+PV)/2)
C=0

def yield_to_maturity(C,FV,PV,t):
    YTM=(C+(FV-PV)/t)/((FV+PV)/2)
    return YTM

ytm_3_yr = yield_to_maturity(C,F,zcb_price_3_yr,T_1)
print(f'The YTM for the 3-year zero-coupon bond is {ytm_3_yr:.4%}')

# Duration
def zcb_duration(maturity_zcb, YTM):
    macaulay_duration = maturity_zcb
    modified_duration = macaulay_duration / (1 + YTM)
    return macaulay_duration, modified_duration

maturity_zcb = 3  # Time to maturity for the zero-coupon bond
macaulay_duration, modified_duration = zcb_duration(maturity_zcb, ytm_3_yr)

print(f'Macaulay Duration for the 3-year ZCB: {macaulay_duration:.2f} years')
print(f'Modified Duration for the 3-year ZCB: {modified_duration:.2f} years')


# Convexity
def zcb_convexity(maturity_zcb, YTM):
    convexity = maturity_zcb * (maturity_zcb + 1) / (1 + YTM) ** 2
    return convexity

convexity_3_yr = zcb_convexity(maturity_zcb, ytm_3_yr)
print(f'Convexity for the 3-year ZCB: {convexity_3_yr:.4f}')


# 2b
# Price

spot_rates = {
    1: 0.035,
    2: 0.045,
    3: 0.0525,
    4: 0.0625
}

coupon_rate_b = 0.05
maturity_b = 2

def coupon_bond_price(F, coupon_rate, maturity, spot_rates):
    C = coupon_rate * F
    price = 0
    for t in range(1, maturity + 1):
        price += C / (1 + spot_rates[t]) ** t
    price += F / (1 + spot_rates[maturity]) ** maturity
    return price

price_b = coupon_bond_price(F, coupon_rate_b, maturity_b, spot_rates)
print(f'The price of the 2-year bond with a 5% coupon is ${price_b:.2f}')

# YTM
C_b = coupon_rate_b * F
ytm_b = yield_to_maturity(C_b, F, price_b, maturity_b)
print(f'The YTM for the 2-year bond with a 5% coupon is {ytm_b:.4%}')


# Duration
def coupon_bond_duration(F, coupon_rate, maturity, YTM, spot_rates):
    C = coupon_rate * F  # Coupon payment
    macaulay_duration = 0
    for t in range(1, maturity + 1):
        macaulay_duration += (t * C) / (1 + spot_rates[t]) ** t
    macaulay_duration += (maturity * F) / (1 + spot_rates[maturity]) ** maturity
    macaulay_duration /= coupon_bond_price(F, coupon_rate, maturity, spot_rates)
    modified_duration = macaulay_duration / (1 + YTM)
    return macaulay_duration, modified_duration

macaulay_duration_b, modified_duration_b = coupon_bond_duration(F, coupon_rate_b, maturity_b, ytm_b, spot_rates)
print(f'Macaulay Duration for the 2-year bond: {macaulay_duration_b:.2f} years')
print(f'Modified Duration for the 2-year bond: {modified_duration_b:.2f} years')


# Convexity
def coupon_bond_convexity(F, coupon_rate, maturity, YTM, spot_rates):
    C = coupon_rate * F  # Coupon payment
    convexity = 0
    for t in range(1, maturity + 1):
        convexity += (t * (t + 1) * C) / (1 + spot_rates[t]) ** (t + 2)
    convexity += (maturity * (maturity + 1) * F) / (1 + spot_rates[maturity]) ** (maturity + 2)
    convexity /= coupon_bond_price(F, coupon_rate, maturity, spot_rates)
    return convexity

convexity_b = coupon_bond_convexity(F, coupon_rate_b, maturity_b, ytm_b, spot_rates)
print(f'Convexity for the 2-year bond: {convexity_b:.4f}')


# 2c
# Price
coupon_rate_c = 0.06
maturity_c = 4

price_c = coupon_bond_price(F, coupon_rate_c, maturity_c, spot_rates)
print(f'The price of the 4-year bond with a 6% coupon is ${price_c:.2f}')

# YTM
C_c = coupon_rate_c * F
ytm_c = yield_to_maturity(C_c, F, price_c, maturity_c)
print(f'The YTM for the 4-year bond with a 6% coupon is {ytm_c:.4%}')

# Duration
macaulay_duration_c, modified_duration_c = coupon_bond_duration(F, coupon_rate_c, maturity_c, ytm_c, spot_rates)
print(f'Macaulay Duration for the 4-year bond: {macaulay_duration_c:.2f} years')
print(f'Modified Duration for the 4-year bond: {modified_duration_c:.2f} years')

# Convexity
convexity_c = coupon_bond_convexity(F, coupon_rate_c, maturity_c, ytm_c, spot_rates)
print(f'Convexity for the 4-year bond: {convexity_c:.4f}')


# 3a
r_1yr = 0.0525
r_2yr = 0.055
bank_forward_rate = 0.056

def forward_rate(r1, r2, t1, t2):
    f = ((1 + r2) ** t2 / (1 + r1) ** t1) - 1
    return f


f_1_2 = forward_rate(r_1yr, r_2yr, 1, 2)

print(f"Market-implied forward rate from year 1 to year 2: {f_1_2:.4%}")
print(f"Bank's offered forward rate: {bank_forward_rate:.4%}")

if f_1_2 > bank_forward_rate:
    print("The bank's forward rate is competitive.")
else:
    print("The market-implied forward rate is higher, so the bank's rate is not competitive.")

# 3b
r_3yr = 0.0575  # 1 & 2 yr already defined in 3a

def strategy_for_investment():
    print("Investment Strategy:")

    if f_1_2 > r_1yr and f_1_2 > r_2yr:
        print(f"The forward rate of {f_1_2:.4%} is competitive, so you can lock it in.")
    else:
        print(
            f"Instead of locking in the forward rate, it may be better to directly invest at the 2-year rate of {r_2yr:.4%}.")

    if r_3yr > r_2yr:
        print(
            f"For a longer-term investment, the 3-year rate of {r_3yr:.4%} offers a better yield than the 2-year rate.")


strategy_for_investment()

# 4a
import pandas as pd
import numpy as np
raw_data = pd.read_csv('DGS10.csv') #ensure csv file is in same directory as .py file
raw_data['DATE'] = pd.to_datetime(raw_data['DATE'], format='%m/%d/%y')
raw_data['IR'] = pd.to_numeric(raw_data['IR'], errors='coerce')
# print(raw_data.head())
clean_data = raw_data.dropna(subset=['IR'])
clean_data = clean_data.sort_values(by='DATE').reset_index(drop=True)
print(clean_data.head())

# interest_rates_pre21 = clean_data[clean_data['DATE'].dt.year < 2021]
interest_rates_pre21 = clean_data[clean_data['DATE'].dt.year < 2021].copy()
interest_rates_post21 = clean_data[clean_data['DATE'].dt.year >= 2021].copy()

# OLS (Linear Regression) Form: 𝑟𝑡 − 𝑟𝑡−1 = 𝑎𝑏 − 𝑎𝑟𝑡−1 + 𝜀𝑡 (0, 𝜎2)

interest_rates_pre21['Delta'] = interest_rates_pre21['IR'].diff()
interest_rates_pre21['Lag'] = interest_rates_pre21['IR'].shift(1)

interest_rates_pre21 = interest_rates_pre21.dropna(subset=['Delta', 'Lag']).reset_index(drop=True)
print(interest_rates_pre21)
# interest_rates_pre21['Delta'] = interest_rates_pre21['IR'].diff()
# interest_rates_pre21.loc[:, 'Delta'] = interest_rates_pre21['IR'].diff()
# interest_rates_pre21.loc[:, 'Lag'] = interest_rates_pre21['IR'].shift(1)
# interest_rates_pre21['Lag'] = interest_rates_pre21['IR'].shift(1)

# interest_rates_pre21 = interest_rates_pre21.dropna()
# interest_rates_pre21 = interest_rates_pre21.dropna(subset=['Delta', 'Lag']).reset_index(drop=True)
# print(interest_rates_pre21)
#
X = interest_rates_pre21['Lag']
Y = interest_rates_pre21['Delta']
mean_X = X.mean()
mean_Y = Y.mean()
cov_XY = np.sum((X - mean_X) * (Y - mean_Y))
var_X = np.sum((X - mean_X) ** 2)

a = cov_XY / var_X
b = mean_Y - a * mean_X

residuals = Y - (a * X + b)
sigma = np.sqrt(np.sum(residuals ** 2) / len(residuals))
print(f'Calibrated Parameters: a = {a}, b = {b}, σ = {sigma}')

#4b
forecasted_values = []
forecasted_volatility = []


for i, row in interest_rates_post21.iterrows():
    lag_value = row['IR']
    forecasted_value = a * (b - lag_value) + lag_value
    forecasted_values.append(forecasted_value)
    forecasted_volatility.append(sigma)

interest_rates_post21['Forecasted_IR'] = forecasted_values
interest_rates_post21['Forecasted_Volatility'] = forecasted_volatility
print(interest_rates_post21[['DATE', 'IR', 'Forecasted_IR', 'Forecasted_Volatility']])

#4c
interest_rates_post21['Forecasted_IR'] = forecasted_values
interest_rates_post21['Actual_IR'] = interest_rates_post21['IR']

mse = np.mean((interest_rates_post21['Forecasted_IR'] - interest_rates_post21['Actual_IR']) ** 2)
ss_total = np.sum((interest_rates_post21['Actual_IR'] - interest_rates_post21['Actual_IR'].mean()) ** 2)
ss_residual = np.sum((interest_rates_post21['Forecasted_IR'] - interest_rates_post21['Actual_IR']) ** 2)
r_squared = 1 - (ss_residual / ss_total)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r_squared}')
