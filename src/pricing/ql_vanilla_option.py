
#%%
import QuantLib as ql
from src.pricing.blackscholes import bsprice

def ql_vanilla_option(S, K, r, q, maturity, vol, evalDate, option_flag, dc=ql.ActualActual(ql.ActualActual.ISMA)):
    ql.Settings.evaluationDate = evalDate
    
    #Vanilla Option
    optionType = ql.Option.Call if option_flag.lower()=="call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(optionType, K)
    euExercise = ql.EuropeanExercise(maturity)
    vanillaOption = ql.VanillaOption(payoff, euExercise)

    #Market
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(S))
    flatRateTs = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    flatDivTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    flatVolTs = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, dc))
    bsm = ql.BlackScholesMertonProcess(spotHandle, flatDivTS, flatRateTs, flatVolTs)
    analyticEngine = ql.AnalyticEuropeanEngine(bsm)

    #Pricing
    vanillaOption.setPricingEngine(analyticEngine)
    price = vanillaOption.NPV()
    return price

if __name__=="__main__":
    S, K, r, q, vol = 100, 100, 0.03, 0.02, 0.2
    TTM = "1M"
    today = ql.Date().todaysDate()
    maturity = today + ql.PeriodParser.parse(TTM)
    dc = ql.ActualActual(ql.ActualActual.ISMA)
    
    option_flag = "put"

    ql_price = ql_vanilla_option(S, K, r, q, maturity, vol, today, option_flag)
    print(f"Option Price = {ql_price:0.8f}")

    T = dc.yearFraction(today, maturity)
    price = bsprice(S, K, r, q, T, vol, option_flag)
    print(f"Option Price = {price:0.8f}")

# %%
