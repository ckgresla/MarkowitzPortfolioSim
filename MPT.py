# Packages Required
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.optimize as sco

from pandas import DataFrame
import pandas as pd




# Markowitz Portfolio Object -- SubClass of the PandasFrame
# class MarkowitzPortfolioDF(DataFrame):

#     # Init Object
#     def __init__(self, *args, **kw):
#         super(MyDF, self).__init__(*args, **kw)
#         if len(args) == 1 and isinstance(args[0], MyDF):
#             args[0]._copy_attrs(self)

#     def _copy_attrs(self, df):
#         for attr in self._attributes_.split(","):
#             df.__dict__[attr] = getattr(self, attr, None)


#     # Extend Pretty Pandas Names
#     @property
#     def _constructor(self):
#         def f(*args, **kw):
#             df = MyDF(*args, **kw)
#             self._copy_attrs(df)
#             return df
#         return f




# Functional API -- Uses a Base Pandas DF

# Multi-Reading dfs -- Convert Data to Input for MPT Simulation
# def generateReturnsTable(table_list):
# for table in table_list: 
#     df = pd.read_csv(table, index_col=None, names=colnames) #iterative read, will subset later

#     # Convert to PandasFrame, Compute Return & Drop Columns
#     df["return"] = ((df["close"]-df["open"])/df["open"])*100 #return for 1m period, in percentage
#     df = df[["open_time", "return"]] #filter out the remaining columns

#     # Add Ticker Symbol (for differentiating in Aggregated df)
#     crypto_name = table.split("-")
#     df["name"] = crypto_name[0]
    
#     # Append to Group Table -- Returns per Crypto for Year
#     if table == "adausdt_hist":
#         dfr = df 
#     else:
#         dfr = pd.concat([dfr, df], axis=0, ignore_index=True)


def annual_performance(weights, mean_returns, cov_matrix, n_trading_days=252):
    """ Function calculates performance for a whole year (can supply less than 1y returns)
    Attributes:
        n_trading_days = [253 or 365] -- depending on whether assessing Stock Market assets or Cryptos (trading days in year)
    """    
    returns = np.sum(mean_returns*weights) * n_trading_days
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(n_trading_days)
    return std, returns

  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(cov_matrix)) #num of assets in users portfolio
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_std_dev, portfolio_return = annual_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return-risk_free_rate) / portfolio_std_dev #excess return or risk adj. = return divided by std dev associated with that return level
    return results, weights_record


# Returns Along Efficient Frontier -- involves optimization call
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns, cov_matrix) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return annual_performance(weights, mean_returns, cov_matrix)[0]

def portfolio_return(weights, mean_returns, cov_matrix):
    return annual_performance(weights, mean_returns, cov_matrix)[1]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Big Kahuna
def SimulatePortfoliosAndPlot(table, num_portfolios=500, rfr=0.0135, plot_efficient_frontier=False):
    
    # Calc Vars Based on Table
    returns = table.pct_change() #calculated based on pandas frame
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = rfr #ad-hoc average unless otherwise specified

    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualized Return:", round(rp, 2))
    print("Annualized Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualized Return:", round(rp_min, 2))
    print("Annualized Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)


    # Plot Majority of Portfolios
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='RdBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()

    # Plot Points for MSR & MV
    plt.scatter(sdp, rp, marker='.', color='#6D8AFF', s=500, label='Maximum Sharpe Ratio')
    plt.scatter(sdp_min, rp_min, marker='.', color='#FF6D6D', s=500, label='Minimum Volatility')
    
    # Plot Efficient Frontier Line -- greatly increases computation time 
    if plot_efficient_frontier:
        target = np.linspace(rp_min, 0.32, 50)
        efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
        # Misc Labels
        plt.title(f'{num_portfolios} - Simulated Portfolios')
        plt.xlabel('annualised volatility')
        plt.ylabel('annualised returns')
        plt.legend(labelspacing=0.8)
        plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-', color='black', label='efficient frontier')

    else:
        # Misc Labels
        plt.title(f'{num_portfolios} - Simulated Portfolios')
        plt.xlabel('annualised volatility')
        plt.ylabel('annualised returns')
        plt.legend(labelspacing=0.8)
    
    return max_sharpe_allocation, min_vol_allocation


# Simulate for Optimal Portfolio
def OptimalPortfolioSim(t, num_portfolios=500, rfr=0.0135):
    # Calc Vars Based on Table
    returns = t.pct_change() #calculated based on pandas frame
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    #num_portfolios = num_portfolios #number of random simulations to run
    risk_free_rate = rfr #ad-hoc average unless otherwise specified
    
    MSR_metrics = {"std_dev":0, "adj_return":0}
    MV_metrics = {"std_dev":0, "adj_return":0}

    # Generate Random Weights per N in num_portfolios
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    
    # Calc Maximum Sharpe ratio Portfolio (best return for risk possible)
    max_sharpe_idx = np.argmax(results[2]) #index of MSR
    #MSR (ratio is the portfolio with the highest risk-adjusted return, i.e best potential reward given the inherrent risk)
    MSR_metrics["std_dev"], MSR_metrics["adj_return"] = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    # max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=t.columns, columns=['allocation'])
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=t.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*1, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    # Calc Minimum Volatility Portfolio
    min_vol_idx = np.argmin(results[0]) #index of least volatile portfolio
    MV_metrics["std_dev"], MV_metrics["adj_return"] = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=t.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*1, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    return max_sharpe_allocation, MSR_metrics, min_vol_allocation, MV_metrics
