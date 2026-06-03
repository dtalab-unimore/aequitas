<details>
<summary><strong>Primitive microstructure features</strong></summary>

<br>

| # | Variable | Description | Source |
|---:|---|---|---|
| 1 | **Close** | Closing price of the asset | EODHD |
| 2 | **Open** | Opening price of the asset | EODHD |
| 3 | **High** | Highest price of the asset during the trading day | EODHD |
| 4 | **Low** | Lowest price of the asset during the trading day | EODHD |
| 5 | **Volume** | Number of traded shares during the trading day | EODHD |
| 6 | **Adj. Close** | Adjusted closing price accounting for corporate actions | EODHD |
| 7 | **Primary Ticker** | Primary ticker symbol used by the data provider | EODHD |
| 8 | **Name** | Official name of the company or financial instrument | EODHD |
| 9 | **GICS Sector** | Industry Classification Standard sector | EODHD |
| 10 | **GICS Group** | GICS industry group classification | EODHD |
| 11 | **GICS Industry** | GICS industry classification | EODHD |
| 12 | **GICS Sub-Industry** | GICS sub-industry classification | EODHD |
| 13 | **Shares Outstanding** | Number of common shares outstanding | EODHD |
| 14 | **Total Equity** | Total shareholders' equity | EODHD |
| 15 | **Total Assets** | Total assets | EODHD |
| 16 | **Total Liabilities** | Total liabilities | EODHD |
| 17 | **Dividends Paid** | Cash dividends paid during the reporting period | EODHD |
| 18 | **Net Income** | Net income reported in the cash flow statement | EODHD |
| 19 | **Total Revenue** | Total revenue | EODHD |
| 20 | **Gross Profit** | Gross profit | EODHD |

</details>

<details>
<summary><strong>Primitive macrostructure features</strong></summary>

<br>

| # | Variable | Description | Source |
|---:|---|---|---|
| 1 | **GSPC** | S&P 500 index level | EODHD |
| 2 | **IXIC** | NASDAQ Composite index level | EODHD |
| 3 | **DJI** | Dow Jones Industrial Average index level | EODHD |
| 4 | **NYA** | NYSE Composite index level | EODHD |
| 5 | **HSI** | Hang Seng Index level | EODHD |
| 6 | **000001.SHG** | SSE Composite Index level | EODHD |
| 7 | **FCHI** | CAC 40 Index level | EODHD |
| 8 | **GDAXI** | DAX Index level | EODHD |
| 9 | **IWM** | iShares Russell 2000 ETF | EODHD |
| 10 | **ISF** | iShares FTSE 100 ETF | EODHD |
| 11 | **USDJPY** | US Dollar / Japanese Yen exchange rate | EODHD |
| 12 | **USDGBP** | US Dollar / British Pound exchange rate | EODHD |
| 13 | **USDCAD** | US Dollar / Canadian Dollar exchange rate | EODHD |
| 14 | **USDCNY** | US Dollar / Chinese Yuan exchange rate | EODHD |
| 15 | **USDAUD** | US Dollar / Australian Dollar exchange rate | EODHD |
| 16 | **USDNZD** | US Dollar / New Zealand Dollar exchange rate | EODHD |
| 17 | **USDCHF** | US Dollar / Swiss Franc exchange rate | EODHD |
| 18 | **USDEUR** | US Dollar / Euro exchange rate | EODHD |
| 19 | **XAUUSD** | Gold spot price in USD | EODHD |
| 20 | **XAGUSD** | Silver spot price in USD | EODHD |
| 21 | **DXY** | US Dollar Index | EODHD |
| 22 | **XOM** | Exxon Mobil Corp stock price | EODHD |
| 23 | **JPM** | JPMorgan Chase & Co stock price | EODHD |
| 24 | **AAPL** | Apple Inc stock price | EODHD |
| 25 | **MSFT** | Microsoft Corp stock price | EODHD |
| 26 | **GE** | General Electric stock price | EODHD |
| 27 | **JNJ** | Johnson & Johnson stock price | EODHD |
| 28 | **WFC** | Wells Fargo stock price | EODHD |
| 29 | **AMZN** | Amazon.com stock price | EODHD |
| 30 | **DTB4WK** | 4-Week Treasury Bill secondary market rate | FRED |
| 31 | **DTB3** | 3-Month Treasury Bill secondary market rate | FRED |
| 32 | **DTB6** | 6-Month Treasury Bill secondary market rate | FRED |
| 33 | **DGS3MO** | 3-Month Treasury constant maturity rate | FRED |
| 34 | **DGS6MO** | 6-Month Treasury constant maturity rate | FRED |
| 35 | **DGS1** | 1-Year Treasury constant maturity rate | FRED |
| 36 | **DGS5** | 5-Year Treasury constant maturity rate | FRED |
| 37 | **DGS10** | 10-Year Treasury constant maturity rate | FRED |
| 38 | **DAAA** | Moody's seasoned Aaa corporate bond yield | FRED |
| 39 | **DBAA** | Moody's seasoned Baa corporate bond yield | FRED |
| 40 | **DCOILWTICO** | WTI crude oil spot price | FRED |
| 41 | **DCOILBRENTEU** | Brent crude oil spot price, Europe | FRED |
| 42 | **HG_Return** | Daily return of Copper futures | Yahoo Finance |
| 43 | **NG_Return** | Daily return of Natural Gas futures | Yahoo Finance |
| 44 | **GC_Return** | Daily return of Gold futures | Yahoo Finance |
| 45 | **SI_Return** | Daily return of Silver futures | Yahoo Finance |

</details>

<details>
<summary><strong>Derivative market microstructure indicators</strong></summary>

<br>

<table>
<thead>
<tr>
<th>#</th>
<th>Variable</th>
<th>Description</th>
<th>Calculation</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>Alpha158</strong></td>
<td>Alpha factor indicators</td>
<td>See Qlib</td>
</tr>
<tr>
<td>2</td>
<td><strong>Alpha360</strong></td>
<td>Alpha factor indicators</td>
<td>See Qlib</td>
</tr>
<tr>
<td>3</td>
<td><strong>c<sub>p</sub></strong></td>
<td><em>p</em> price normalized by close price minus one, where <em>p</em> ∈ {Open, High, Low}</td>
<td><code>c_p = p / Close - 1</code></td>
</tr>
<tr>
<td>4</td>
<td><strong>p_price_change</strong></td>
<td>Daily <em>p</em> price return, where <em>p</em> ∈ {Open, High, Low, Close}</td>
<td><code>p_price_change = p_t / p_{t-1} - 1</code></td>
</tr>
<tr>
<td>5</td>
<td><strong>adj_p_price_change</strong></td>
<td>Daily adjusted <em>p</em> price return, where <em>p</em> ∈ {Open, High, Low, Close}</td>
<td><code>adj_p_price_change = Adj.p_t / Adj.p_{t-1} - 1</code></td>
</tr>
<tr>
<td>6</td>
<td><strong>p_close_ratio_d1</strong></td>
<td><em>p</em> price to previous close ratio, where <em>p</em> ∈ {Open, High, Low, Close}</td>
<td><code>p_close_ratio_d1 = p_t / Close_{t-1}</code></td>
</tr>
<tr>
<td>7</td>
<td><strong>adj_p_close_ratio_d1</strong></td>
<td>Adjusted <em>p</em> price to previous adjusted close ratio, where <em>p</em> ∈ {Open, High, Low, Close}</td>
<td><code>adj_p_close_ratio_d1 = Adj.p_t / Adj.Close_{t-1}</code></td>
</tr>
<tr>
<td>8</td>
<td><strong>adj_close_abs_return</strong></td>
<td>Absolute adjusted close price change</td>
<td><code>Adj.Close_t - Adj.Close_{t-1}</code></td>
</tr>
<tr>
<td>9</td>
<td><strong>close_abs_return</strong></td>
<td>Absolute close price change</td>
<td><code>Close_t - Close_{t-1}</code></td>
</tr>
<tr>
<td>10</td>
<td><strong>z<sub>dk</sub></strong></td>
<td>Normalized k-day adjusted close moving average, k ∈ {5, 10, 15, 20, 25, 30}</td>
<td><code>sum_{i=0}^{k-1} Adj.Close_{t-i} / (k * Adj.Close_t) - 1</code></td>
</tr>
<tr>
<td>11</td>
<td><strong>f<sub>dk</sub></strong></td>
<td>Alternative normalized k-day adjusted close moving average, k ∈ {5, 10, 15, 20, 25, 30}</td>
<td><code>sum_{i=0}^{k-1} Adj.Close_{t-i} / (k * (Adj.Close_t - 1))</code></td>
</tr>
<tr>
<td>12</td>
<td><strong>vol</strong></td>
<td>Daily volume return</td>
<td><code>Volume_t / Volume_{t-1} - 1</code></td>
</tr>
<tr>
<td>13</td>
<td><strong>mom<sub>k</sub></strong></td>
<td>k-day adjusted close momentum, k ∈ {1, 2, 3}</td>
<td><code>Adj.Close_t - Adj.Close_{t-k}</code></td>
</tr>
<tr>
<td>14</td>
<td><strong>roc<sub>k</sub></strong></td>
<td>k-day rate of change of adjusted close, k ∈ {5, 10, 15, 20}</td>
<td><code>100 * (Adj.Close_t - Adj.Close_{t-k}) / Adj.Close_{t-k}</code></td>
</tr>
<tr>
<td>15</td>
<td><strong>ema<sub>k</sub></strong></td>
<td>k-day exponential moving average of adjusted close, k ∈ {10, 20, 50, 200}</td>
<td><code>EMA(Adj.Close, k)</code></td>
</tr>
<tr>
<td>16</td>
<td><strong>Adj. p</strong></td>
<td><em>p</em> price adjusted using the adjustment factor, where <em>p</em> ∈ {Open, High, Low}</td>
<td><code>Adj.p = p * Adj.Close / Close</code></td>
</tr>
<tr>
<td>17</td>
<td><strong>TTM Dividends Paid</strong></td>
<td>Trailing twelve months of cash dividends paid</td>
<td><code>sum_{i=0}^{3} DividendsPaid_{t-i}</code></td>
</tr>
<tr>
<td>18</td>
<td><strong>TTM Net Income</strong></td>
<td>Trailing twelve months of net income</td>
<td><code>sum_{i=0}^{3} NetIncome_{t-i}</code></td>
</tr>
<tr>
<td>19</td>
<td><strong>TTM Total Revenue</strong></td>
<td>Trailing twelve months of total revenue</td>
<td><code>sum_{i=0}^{3} TotalRevenue_{t-i}</code></td>
</tr>
<tr>
<td>20</td>
<td><strong>TTM Gross Profit</strong></td>
<td>Trailing twelve months of gross profit</td>
<td><code>sum_{i=0}^{3} GrossProfit_{t-i}</code></td>
</tr>
<tr>
<td>21</td>
<td><strong>Sales Growth</strong></td>
<td>Growth rate of total revenue over the last year</td>
<td><code>TotalRevenue_t / TotalRevenue_{t-4} - 1</code></td>
</tr>
<tr>
<td>22</td>
<td><strong>Earnings Growth</strong></td>
<td>Growth rate of net income over the last year</td>
<td><code>NetIncome_t / NetIncome_{t-4} - 1</code></td>
</tr>
<tr>
<td>23</td>
<td><strong>Gross Profit to Assets</strong></td>
<td>Ratio of TTM gross profit to total assets</td>
<td><code>TTM GrossProfit / TotalAssets</code></td>
</tr>
<tr>
<td>24</td>
<td><strong>ROE</strong></td>
<td>Return on equity</td>
<td><code>TTM NetIncome / TotalEquity</code></td>
</tr>
<tr>
<td>25</td>
<td><strong>ROA</strong></td>
<td>Return on assets</td>
<td><code>TTM NetIncome / TotalAssets</code></td>
</tr>
<tr>
<td>26</td>
<td><strong>Market Leverage</strong></td>
<td>Total liabilities to total assets ratio</td>
<td><code>TotalLiabilities / TotalAssets</code></td>
</tr>
<tr>
<td>27</td>
<td><strong>Market Cap</strong></td>
<td>Market capitalization</td>
<td><code>Adj.Close * SharesOutstanding</code></td>
</tr>
<tr>
<td>28</td>
<td><strong>Dividend Yield</strong></td>
<td>TTM dividends over market cap</td>
<td><code>TTM DividendsPaid / MarketCap</code></td>
</tr>
<tr>
<td>29</td>
<td><strong>Earnings to Price</strong></td>
<td>TTM net income over market cap</td>
<td><code>TTM NetIncome / MarketCap</code></td>
</tr>
<tr>
<td>30</td>
<td><strong>Sales to Price</strong></td>
<td>TTM total revenue over market cap</td>
<td><code>TTM TotalRevenue / MarketCap</code></td>
</tr>
<tr>
<td>31</td>
<td><strong>Book Value to Price</strong></td>
<td>Total equity over market cap</td>
<td><code>TotalEquity / MarketCap</code></td>
</tr>
<tr>
<td>32</td>
<td><strong>Volatility</strong></td>
<td>Annualized rolling 252-day volatility of adjusted close returns</td>
<td><code>Std(Adj.Close returns, 252)</code></td>
</tr>
<tr>
<td>33</td>
<td><strong>Momentum</strong></td>
<td>21-day vs 252-day price momentum</td>
<td><code>Adj.Close_{t-21} / Adj.Close_{t-252} - 1</code></td>
</tr>
<tr>
<td>34</td>
<td><strong>REV</strong></td>
<td>Recent 1-month return</td>
<td><code>Adj.Close_t / Adj.Close_{t-21} - 1</code></td>
</tr>
<tr>
<td>35</td>
<td><strong>Share Turnover</strong></td>
<td>21-day volume over shares outstanding</td>
<td><code>sum_{i=0}^{20} Volume_{t-i} / SharesOutstanding</code></td>
</tr>
</tbody>
</table>

</details>

<details>
<summary><strong>Derivative market macrostructure features</strong></summary>

<br>

<table>
<thead>
<tr>
<th>#</th>
<th>Variable</th>
<th>Description</th>
<th>Calculation</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><strong>price_change</strong></td>
<td>Daily adjusted close return of market index</td>
<td><code>Adj.Close_t / Adj.Close_{t-1} - 1</code></td>
</tr>
<tr>
<td>2</td>
<td><strong>mean_price_change<sub>k</sub></strong></td>
<td>Rolling mean of daily adjusted close returns, k ∈ {5, 10, 20, 30, 60}</td>
<td><code>(1 / k) * sum_{i=0}^{k-1} price_change_{t-i}</code></td>
</tr>
<tr>
<td>3</td>
<td><strong>std_price_change<sub>k</sub></strong></td>
<td>Rolling standard deviation of daily adjusted close returns, k ∈ {5, 10, 20, 30, 60}</td>
<td><code>sqrt((1 / k) * sum_{i=0}^{k-1} (price_change_{t-i} - mu_k)^2)</code></td>
</tr>
<tr>
<td>4</td>
<td><strong>mean_vol<sub>k</sub></strong></td>
<td>Rolling mean of trading volume normalized by current volume, k ∈ {5, 10, 20, 30, 60}</td>
<td><code>((1 / k) * sum_{i=0}^{k-1} Volume_{t-i}) / Volume_t</code></td>
</tr>
<tr>
<td>5</td>
<td><strong>std_vol<sub>k</sub></strong></td>
<td>Rolling standard deviation of trading volume normalized by current volume, k ∈ {5, 10, 20, 30, 60}</td>
<td><code>sqrt((1 / k) * sum_{i=0}^{k-1} (Volume_{t-i} - Vbar_k)^2) / Volume_t</code></td>
</tr>
<tr>
<td>6</td>
<td><strong>Commodity_Return<sub>c</sub></strong></td>
<td>Daily return of commodity futures, c ∈ {HG, NG, GC, SI}</td>
<td><code>P_t^{c} / P_{t-1}^{c} - 1</code></td>
</tr>
<tr>
<td>7</td>
<td><strong>World_Index_Return<sub>i</sub></strong></td>
<td>Daily return of global equity indices, i ∈ {GSPC, IXIC, DJI, NYA, HSI, SSE, FCHI, GDAXI}</td>
<td><code>P_t^{i} / P_{t-1}^{i} - 1</code></td>
</tr>
<tr>
<td>8</td>
<td><strong>ETF_Return<sub>f</sub></strong></td>
<td>Daily return of equity ETFs, f ∈ {IWM, ISF}</td>
<td><code>P_t^{f} / P_{t-1}^{f} - 1</code></td>
</tr>
<tr>
<td>9</td>
<td><strong>FX_Return<sub>x</sub></strong></td>
<td>Daily return of foreign exchange rates and metal FX pairs, x ∈ {USDJPY, USDGBP, USDCAD, USDCNY, USDAUD, USDNZD, USDCHF, USDEUR, XAUUSD, XAGUSD}</td>
<td><code>P_t^{x} / P_{t-1}^{x} - 1</code></td>
</tr>
<tr>
<td>10</td>
<td><strong>USD_Index_Return</strong></td>
<td>Daily return of the US Dollar Index, DXY</td>
<td><code>P_t^{DXY} / P_{t-1}^{DXY} - 1</code></td>
</tr>
<tr>
<td>11</td>
<td><strong>US_Stock_Return<sub>s</sub></strong></td>
<td>Daily return of selected US equities, s ∈ {XOM, JPM, AAPL, MSFT, GE, JNJ, WFC, AMZN}</td>
<td><code>P_t^{s} / P_{t-1}^{s} - 1</code></td>
</tr>
<tr>
<td>12</td>
<td><strong>TE1</strong></td>
<td>Term spread, 10Y minus 4W</td>
<td><code>DGS10_t - DTB4WK_t</code></td>
</tr>
<tr>
<td>13</td>
<td><strong>TE2</strong></td>
<td>Term spread, 10Y minus 3M</td>
<td><code>DGS10_t - DTB3_t</code></td>
</tr>
<tr>
<td>14</td>
<td><strong>TE3</strong></td>
<td>Term spread, 10Y minus 6M</td>
<td><code>DGS10_t - DTB6_t</code></td>
</tr>
<tr>
<td>15</td>
<td><strong>TE5</strong></td>
<td>Term spread, 3M minus 4W</td>
<td><code>DTB3_t - DTB4WK_t</code></td>
</tr>
<tr>
<td>16</td>
<td><strong>TE6</strong></td>
<td>Term spread, 6M minus 4W</td>
<td><code>DTB6_t - DTB4WK_t</code></td>
</tr>
<tr>
<td>17</td>
<td><strong>DE1</strong></td>
<td>Default spread, Baa minus Aaa</td>
<td><code>DBAA_t - DAAA_t</code></td>
</tr>
<tr>
<td>18</td>
<td><strong>DE2</strong></td>
<td>Default spread, Baa minus 10Y</td>
<td><code>DBAA_t - DGS10_t</code></td>
</tr>
<tr>
<td>19</td>
<td><strong>DE4</strong></td>
<td>Default spread, Baa minus 6M</td>
<td><code>DBAA_t - DTB6_t</code></td>
</tr>
<tr>
<td>20</td>
<td><strong>DE5</strong></td>
<td>Default spread, Baa minus 3M</td>
<td><code>DBAA_t - DTB3_t</code></td>
</tr>
<tr>
<td>21</td>
<td><strong>DE6</strong></td>
<td>Default spread, Baa minus 4W</td>
<td><code>DBAA_t - DTB4WK_t</code></td>
</tr>
</tbody>
</table>

</details>
