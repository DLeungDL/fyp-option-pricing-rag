# Option pricing with neural networks vs. Black-Scholes under different volatility forecasting approaches for BIST 30 index options

> 原文檔案：`Option_pricing_with_neural_networks_vs._Black-Scholes_under_different_volatility.md`  
> 語言：繁體中文（臺灣，zh-TW）  
> 說明：由 LlamaParse Markdown 分段機器翻譯；公式、表格數字與檔名未改寫。

---

**Full Length Article**

# 以神經網路對 BIST 30 指數選擇權進行選擇權定價：與 Black-Scholes 在不同波動率預測方法下的比較

Zeynep İltüzer

*伊斯坦堡艾凡薩雷大學（Istanbul Ayvansaray University），經濟、行政與社會科學學院，企業管理系，Prof. Muammer Aksoy Cad. No: 10 Kazlıçeşme/Zeytinburnu, İstanbul, Turkiye*

收稿日期：2021 年 8 月 26 日；修訂日期：2021 年 12 月 26 日；接受日期：2021 年 12 月 27 日  
線上發表日期：2021 年 12 月 31 日

---

### 摘要

本研究比較神經網路（neural network）與 Black-Scholes 模型在不同波動率預測方法下，對 BIST30（Borsa Istanbul）指數買權與賣權進行定價的表現。由於波動率是選擇權定價的關鍵參數，本文採用 GARCH（廣義自迴歸條件異方差，Generalized Autoregressive Conditional Heteroskedasticity）、隱含波動率（implied volatility）、歷史波動率（historical volatility）以及隱含波動率指數（VBI）來決定最適合的波動率方法，並依據價內程度（moneyness）與剩餘到期時間（time-to-maturity）兩個維度進行評估。本文亦包含子樣本分析，檢視模型在動盪時期（turbulent periods）的定價表現。整體結果顯示，在平穩時期（tranquil times），神經網路對買權的定價表現優於 Black-Scholes；而在動盪時期，Black-Scholes 則優於神經網路。對於賣權而言，在平穩時期 Black-Scholes 模型表現最佳，而在動盪時期則以神經網路表現最佳。
Copyright © 2021, Borsa İstanbul Anonim Şirketi. Published by Elsevier B.V. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).

*JEL 分類：* C45; G13; G14

*關鍵詞：* BIST 指數選擇權；Black-Scholes；神經網路（Neural network）；波動率（Volatility）

---

## 1. 緒論

作為國際金融市場的自由化與相互連結程度提高，經濟主體所面臨的風險也隨之增加並快速變化。金融衍生性商品的訂價，尤其是用來管理並在這些日益波動的市場中求存的選擇權，其重要性日益提升，也促使相關文獻與實務領域快速發展。雖然最早關於選擇權訂價的研究可追溯至1900年代初期，但Black與Scholes（BS；1972）的開創性論文已成為選擇權訂價文獻與選擇權交易的基石，因為它被金融市場的實務人士廣泛接受並使用。自此之後，許多研究致力於放寬該模型不切實際的假設，例如Cox et al. (1979)、Rendleman and Bartter (1979)、Rubinstein (1983)、Boyle (1988)、Hull and White (1987)、Scott (1987)、Naik (1993)、Amin and Ng (1993)、Duan (1995)以及Scott (2002)等。標的資產波動率為常數的假設，被許多分析BS模型錯誤訂價的研究（如Macbeth and Merville (1979)、Dumas et al. (2002)以及Poon (2005)）視為最重要的假設，必須加以放寬才能獲得更精確的訂價公式。對於參數模型而言，波動率如何被建模——例如連續隨機過程或跳躍擴散過程——對於模型是否成功扮演關鍵角色。然而，這也大幅提高了模型的數學複雜度，限制了大多數實務人士的理解與使用。在發展出許多不同版本的BS選擇權訂價模型以處理該模型各種假設之後，使用與測試人工神經網路（artificial neural networks, NNs）於選擇權訂價，已吸引了廣泛的關注。

Z. İltüzer

*Borsa İstanbul Review 22-4 (2022) 725–742*

被金融領域研究者視為一種不需要對變數及其關係做出任何假設的替代性定價模型。<sup>1</sup>

神經網路（NNs）是一種機器學習技術，由於資料可用性的增加以及硬體與軟體技術的發展，在過去20年間已被廣泛應用於許多學科與產業。此領域的多數研究均提供強烈證據，顯示NNs的表現優於布萊克-休斯模型（BS model）。<font color="blue">Malliaris and Salchenberger (1993)</font>、<font color="blue">Hutchinson et al. (1994)</font>、<font color="blue">Gradojevic et al. (2009)</font>、<font color="blue">Garcia and Gencay (2000)</font>、<font color="blue">Yang et al. (2017)</font>以及<font color="blue">Fadda (2020)</font>將NNs與BS模型在S&P指數選擇權或期貨選擇權上的表現進行比較，均得出NNs優於BS模型的結論，但其中少數研究發現BS模型在短期價內選擇權的表現較佳。NNs在選擇權定價上的成功，也見於其他成熟股市指數選擇權。<font color="blue">Yao et al. (2000)</font>以NNs和BS模型檢視日經225指數選擇權或期貨選擇權的定價，證據顯示NNs在價內與價外選擇權的定價表現較佳，而BS模型則在價平選擇權的定價上表現較優。<font color="blue">Amilon (2003)</font>分析NNs與BS模型在瑞典股市指數選擇權上的表現，兩種模型均使用隱含波動率與歷史波動率估計值作為波動率輸入，結果提供強烈證據，顯示使用隱含波動率的NNs在所有價內外程度均有較佳表現。<font color="blue">Anders et al. (1998)</font>對DAX指數選擇權進行比較分析，應用統計推論技術來決定最佳NN架構，並報告NNs優於BS模型。<font color="blue">Bennell and Sutcliffe (2005)</font>比較NNs與BS模型在FTSE 100指數選擇權上的表現，得出與S&P指數選擇權研究類似的結論，即NNs在價平與價外選擇權的定價上表現較優，但BS模型在價內選擇權的定價上表現較佳。據本人所知，<font color="blue">Lin and Yeh (2005)</font>是唯一針對新興股市指數選擇權，比較NNs與BS模型定價表現的研究。他們的發現顯示，BS模型在台灣股市指數選擇權的定價上優於NN模型。<font color="blue">Wang (2009a)</font>、<font color="blue">Lin and Yeh (2009)</font>、<font color="blue">Tseng et al. (2008)</font>以及<font color="blue">Wang et al. (2012)</font>則在不同波動率估計方法下（例如歷史波動率、隱含波動率，以及對稱與非對稱GARCH（廣義自迴歸條件異變異數）波動率），以NNs檢視台灣股市指數選擇權的定價，但未與BS模型進行比較。其中部分研究指出GARCH家族模型能提升NNs的定價表現，其他研究則顯示隱含波動率方法提供較佳的表現。

**輸出：**

總體而言，文獻顯示在為選擇權定價時，類神經網路（neural networks, NNs）相較於 Black-Scholes（BS）模型表現更優異，特別是在成熟股市指數的價平（at-the-money）與價外（out-of-the-money）選擇權上。然而，由於針對新興股市選擇權的研究數量甚少，目前尚不清楚 NNs 的優異表現是否也適用於新興股市的選擇權。本研究透過比較類神經網路與 BS 模型在 BIST < Borsa Istanbul > 30 指數選擇權（call 與 put）的定價表現，填補了此一研究缺口。據本人所知，這是第一篇針對土耳其股市指數選擇權進行類神經網路定價，並將其表現與傳統 BS 模型進行比較的研究。由於波動率（volatility）是選擇權定價的關鍵輸入變數，本研究亦探討使用不同波動率預測方法——亦即隱含波動率（implied volatility）、短期與長期歷史波動率（historical volatility），以及 GARCH 波動率——對模型定價表現的影響。更具體而言，本研究試圖回答以下問題：非參數的人工類神經網路模型或傳統 BS 模型，何者在 BIST 30 指數選擇權的定價上表現較佳？哪些波動率預測方法能夠提升模型的定價表現，並如何影響模型的定價行為？此外，模型的表現是否會因選擇權的價內程度（moneyness）與剩餘到期時間（time to maturity）而有所不同？找出最適合 call 與 put 選擇權的定價模型，將有助於針對市場上被低估與高估的選擇權，採用不同的交易策略，從而使避險、投資組合配置與風險管理決策更為有效。

第 **2** 節提供模型的簡要說明、方法論細節以及資料描述。實證結果呈現於第 **3** 節。第 **4** 節則提出總結與整體結論。

## 2. 模型、方法論與資料

### 2.1. 人工類神經網路（Artificial Neural Network, ANN）

人工類神經網路（ANNs）是一種能夠從樣本資料中學習的資訊處理模型。其中最常使用的類型為多層感知器（multilayer perceptron, MLP）。多層感知器最基本的架構如 <font color="blue">Fig. 1</font> 所示。輸入變數 $X_i$ 被饋入第一層（輸入層），而網路的輸出 $Y_j$ 則由最後一層（輸出層）產生。在輸入層與輸出層之間存在若干隱藏層（hidden layers），其神經元數量需在學習過程中加以確定。

多層感知器架構的基本處理單元為神經元（neuron），每個神經元皆以特定權重 $w$ 與下一層的所有神經元相連接，這意味著多層感知器是全連接（fully connected）的。每個神經元皆使用非線性激活函數 $\varnothing$，將加權後的訊號進行轉換，並傳遞至下一層。透過這種方式，輸入變數 $X_i$ 以權重 $w_{ik}$ 被前饋（fed forward）至隱藏層，而隱藏層中的神經元 $h_k$ 則負責後續的處理。

---
<sup>1</sup> Amilon (2003)、Anders et al. (1998)、Bennell and Sutcliffe (2005)、Daglish (2003)、Fadda (2020)、Garcia and Gencay (2000)、Gaspar et al. (2020)、Gradojevic et al. (2009)、Hutchinson et al. (1994)、İltüzer Samur and Temur (2009)、Ivas,cu (2021)、Lajbcygier (2004)、Lin and Yeh (2005, 2009)、Malliaris and Salchenberger (1993)、Morelli et al. (2004)、Tseng et al. (2008)、Wang (2009a, 2009b)、Wang et al. (2012)、Yadav (2021)、Yang et al. (2017)，以及<font color="blue">Yao et al. (2000)</font>。

726
---

Z. İltizler

Borsa İstanbul Review 22-4 (2022) 725–742

## 圖 1. 單隱藏層 MLP（Multilayer Perceptron）架構

反向傳播誤差  
前向傳遞激活

<table>
  <tr>
    <th>層級</th>
    <th>節點</th>
  </tr>
<tr>
    <td>輸入層</td>
    <td>X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>i</sub></td>
  </tr>
<tr>
    <td>隱藏層</td>
    <td>h<sub>k</sub></td>
  </tr>
<tr>
    <td>輸出層</td>
    <td>Y<sub>1</sub>, Y<sub>i</sub></td>
  </tr>
</table>

如式（1）所示，經過加權後的前向傳遞至輸出層 $Y_j$，其權重為 $w_{kj}$，如式（2）所示。

$$h_k = \varphi\left(\sum_{i} w_{ik} x_i + b_1\right) \tag{1}$$

$$y_j = \varphi\left(\sum_{k} w_{kj} h_k + b_2\right) \tag{2}$$

其中 $b_1$ 與 $b_2$ 分別為隱藏層與輸出層的偏差項（bias term）。本研究於隱藏層與輸出層皆採用修正線性單元（rectified linear unit, ReLU）作為激活函數，其定義如式（3）所示。ReLU 是近年來在各種神經網路（neural network, NN）應用中最受歡迎的激活函數之一，其表現優於其他常見激活函數，例如雙曲正切（hyperbolic tangent）與 sigmoid 函數（Glorot et al. (2011)；Zaheer & Shaziya, 2018）。

$$\varphi(z) = \max\{0, z\} \tag{3}$$

網路的輸出值 $Y_j$ 會與實際觀測值 $t_j$ 進行比較，透過式（4）計算平方誤差總和，接著將誤差反向傳播，以更新權重 $w_{ik}$ 與 $w_{kj}$，使總誤差達到最小化。

$$L = \frac{1}{2} \sum_{j} (t_j - Y_j)^2 \tag{4}$$

本研究建構兩個神經網路模型，如式（5）與（6）所示。第一個模型採用 Black-Scholes（BS）模型的輸入與輸出，其中 $S_t$ 為時間 $t$ 時的指數現貨價格，$X$ 為選擇權的履約價格，$\sigma_t$ 為時間 $t$ 時的波動率，$r_t$ 為時間 $t$ 時的無風險利率，$T-t$ 為距到期時間，輸出則為選擇權買權價格 $c$（或賣權價格 $p$）。

$$c_t = f(S_t, X, r_t, T-t, g(\sigma_t)) \tag{5}$$

其中 $g(\sigma_t)$ 表示第 2.3 節所詳述的不同波動率預測方法之結果。第二個模型如式（6）所示，係參考 Hutchinson et al. (1994) 的做法，由於 $f$ 對 $X$ 與 $S_t$ 為一次齊次函數（homogeneous of degree one），因此網路將 $\frac{S_t}{X}$ 作為單一輸入，而非將 $S_t$ 與 $X$ 分開輸入，並將輸出映射為選擇權價格除以履約價格，即 $\frac{c_t(p_t)}{X}$。

$$\frac{c_t}{X} = f\left(\frac{S_t}{X}, r_t, T-t, g(\sigma_t)\right) \tag{6}$$

**Output:**

2017 年 3 月至 2021 年 8 月之間的歐式 BIST 30 指數買權與賣權選擇權資料被用於本研究分析。<sup>2</sup> 2016 年 1 月至 2021 年 8 月的 BIST 30 每日收盤價，則用於第 2.3 節所述的波動率估計（在需要時使用）。無風險利率則以最接近選擇權到期日的兩個銀行間拆款利率（TRLIBOR）進行線性內插作為土耳其經濟的近似值。<sup>3</sup> 輸入與輸出分別由式（5）與式（6）所代表的類神經網路（neural networks, NNs），以批次模式（batch mode）進行訓練，並採用有限記憶體 Broyden–Fletcher–Goldfarb–Shanno 演算法（limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm, LBFGS）來最佳化調整網路權重。<sup>4</sup> 模型選擇方面，採用交叉驗證（cross-validation）方法，將資料分割為三部分：訓練集、驗證集與測試集。其中 2017 年 3 月至 2020 年 6 月的資料用於訓練，2020 年 7 月至 12 月的資料用於驗證，2021 年 1 月至 8 月的資料則作為測試集。本研究遵循類似 <font color="#0000FF">Gu et al. (2020)</font> 的做法，將資料分割為訓練、測試與驗證期間。此外，為評估模型在動盪時期的表現，本研究另使用 2017 年 3 月至 2018 年 4 月的子樣本期間。世界不確定性指數（World Uncertainty Index, WUI）中由 <font color="#0000FF">Ahir et al. (2018)</font> 針對土耳其所建構的 WUITUR 資料（2017 年 3 月至 2021 年 8 月），用以判斷全樣本期間的動盪時期。<sup>5</sup> 根據 WUITUR，全樣本期間於 2018 年 4 月出現高峰，因此本研究亦評估模型在 2018 年 4 月的預測能力。此子樣本中，2017 年 3 月至 12 月的資料用於訓練，2018 年 1 月至 3 月的資料用於驗證，2018 年 4 月的資料則作為測試集。

驗證集根均方誤差（root mean squared error）最低的網路架構被選為最適架構，其公式為  
$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (c_i - \widehat{c}_i)^2}$，  
其中 $c_i$ 為買權（賣權）選擇權的收盤價，$\widehat{c}_i$ 為類神經網路模型預測的選擇權價格。測試集不參與網路架構的選擇，僅用於檢驗模型的樣本外表現。為避免過度擬合訓練資料，本研究採用早期停止法（early stopping）。亦即，模型在

---
<sup>2</sup> 選擇權資料取自 <font color="#0000FF">datastore.borsaistanbul.com</font>。  
<sup>3</sup> TRLIBOR 歷史資料取自 <font color="#0000FF">www.trlibor.org</font>。  
<sup>4</sup> 本研究使用 Python 及其相關函式庫建構與估計網路參數。  
<sup>5</sup> WUITUR 資料可於 <font color="#0000FF">https://fred.stlouisfed.org/series/WUITUR/</font> 取得。

**Z. İltızer**

*Borsa İstanbul Review 22-4 (2022) 725–742*

在訓練資料上進行訓練，並在驗證資料上監控效能改善。當驗證集的誤差開始上升時，早期停止法（early stopping）會停止訓練並防止過擬合。

*2.2. Black-Scholes 選擇權定價模型*

$$c_t = S_t N(d_1) - X e^{-r_t (T-t)} N(d_2) \tag{7}$$

$$p_t = X e^{-r_t (T-t)} N(-d_2) - S_t N(-d_1) \tag{8}$$

$$d_1 = \frac{\ln\left(\frac{S_t}{X}\right) + \left(r_t + \frac{g(\sigma_t)^2}{2}\right)(T-t)}{g(\sigma_t)\sqrt{T-t}} \tag{9}$$

$$d_2 = d_1 - g(\sigma_t)\sqrt{T-t} \tag{10}$$

其中 $N(x)$ 為累積機率分配函數，而 $g(\sigma_t)$ 則是來自不同波動率預測方法所得到的波動率估計值，詳細說明於第 <u>2.3</u> 節。

**2.3. 波動率預測**

**2.3.1. 歷史波動率（Historical volatility）**

歷史波動率預測基本上是每日對數報酬率資料 $R_t = \ln\left(\frac{S_t}{S_{t-1}}\right)$ 的年化標準差，如公式 (11)<sup>6</sup> 所示。共估計五種不同版本的歷史波動率，依據所涵蓋過去觀測值的時間長度而定，分別為以日曆日為基礎的 360 天、30 天、10 天，以及以交易日為基礎的 21 天與 252 天，藉此反映股市的短期與相對長期趨勢。

$$\widehat{\sigma}_{t+1} = \sqrt{\frac{1}{n-1} \sum_{i=t-(n-1)}^{i=t} (R_i - \overline{R})^2 \sqrt{360}} \tag{11}$$

其中 $\overline{R}$ 為每日對數報酬率的平均值，$n$ 分別等於 10 天、30 天、360 天、21 天與 252 天，對應五種不同的歷史波動率預測，分別表示為 $g(\sigma_t) = \widehat{\sigma}_t^{10}$、$g(\sigma_t) = \widehat{\sigma}_t^{30}$、$g(\sigma_t) = \widehat{\sigma}_t^{360}$、$g(\sigma_t) = \widehat{\sigma}_t^{21}$ 與 $g(\sigma_t) = \widehat{\sigma}_t^{252}$。選擇 10 天與 30 天資料的依據來自 <u>Amilon (2003)</u> 的研究。此外，本研究亦估計以最近一年資料為基礎的歷史波動率，以檢視當使用考量標的資產相對長期特性的波動率預測時，模型的表現是否有所提升。

**2.3.2. GARCH 模型**

GARCH 模型由 <u>Bollerslev (1986)</u> 提出，該模型考量了股市報酬率中所觀察到的波動率群聚（volatility clustering）與異質變異（heteroskedasticity）現象。本研究採用 GARCH(1,1) 模型，其形式如下：

$$R_t = \mu + \varepsilon_t$$
$$\varepsilon_t = \sigma_t z_t, \ z_t \sim N(0, 1) \tag{12}$$
$$\sigma_t = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

研究中，使用預測日前最近一年的 BIST 30 指數對數報酬率 $R_t$ 來估計公式 <u>(12)</u> 的參數。接著，將該模型用於預測日期 $t$ 的波動率，並乘以 $\sqrt{360}$ 進行年化處理。每一天樣本資料均向前滾動一天，並重複相同程序。GARCH(1,1) 模型的波動率預測為 $g(\sigma_t) = \widehat{\sigma}_t^{garch} = \sigma_t \sqrt{360}$（日期 $t$）。

**2.3.3. 隱含波動率（Implied volatility）**

**Translated Output:**

隱含波動率預測值 $\widehat{\sigma}_t^{implied}$ 是透過校準 Black-Scholes 模型的波動率參數，使其完美符合時間 $t$ 當日平價（at-the-money）買權（賣權）收盤價而獲得，隨後用於當日市場上其他交易選擇權的波動率預測。亦即 $g(\sigma_t) = \widehat{\sigma}_t^{implied}$。當當日 $t$ 無 moneyness 等於 1 的買權（賣權）時，則選擇 moneyness 最接近 1 的選擇權進行校準。文獻中普遍接受平價（ATM）選擇權的隱含波動率為選擇權價格所反映的標的資產真實波動率（<u>Chance et al., 2017</u>）。

**2.3.4. VBI**

由 <u>Sensoy and Omole (2018)</u> 針對土耳其選擇權市場開發的隱含波動率指數（VBI），$\widehat{\sigma}_t^{vbi}$，被用作時間 $t$ 的波動率輸入。他們針對土耳其選擇權市場估計類似 VIX 的指數時，提供了一套參數選擇程序的指引，該程序考量了市場微結構，尤其是與開發市場相比之下土耳其股市相對缺乏流動性的特性，而 VIX 主要是為開發市場所設計。有關估計方法與程序的細節，請參見 <u>Sensoy and Omole (2018)</u>。

## 3. 實證分析與結果

在本節中，我們使用八種不同的波動率預測方法，來比較以方程式 <u>(5) 和 (6)</u> 所代表的類神經網路（NN）模型，以及傳統 Black-Scholes 模型在 BIST 30 指數買權與賣權定價上的表現。模型評估係以樣本外均方根誤差（Root Mean Squared Error, RMSE）為基準，並依選擇權的 moneyness 與剩餘到期時間進行分組。我們執行 Diebold-Mariano 檢定，以檢驗各模型預測準確度差異的統計顯著性。選擇權依其 moneyness（$S_t/K$）分為三類，此分類方式參考 <u>Gradojevic et al. (2009)</u>、<u>Tseng et al. (2008)</u> 與 <u>Lin and Yeh (2009)</u> 的類似做法。買權中 moneyness 介於 0.97 至 1.03 者被歸類為平價（ATM）選擇權（對賣權而言亦為 ATM），moneyness 高於 1.03 者歸類為價內（ITM）選擇權（對賣權而言為價外 OTM），而 moneyness 低於 0.97 者則歸類為價外（OTM）選擇權（對賣權而言為價內 ITM）。依循 <u>Fadda (2020)</u> 的做法，

<sup>6</sup> 以 252 個交易日進行年化計算。

728

根據到期期限維度，到期期限在一個月以內的選擇權被歸類為短期選擇權，到期期限介於一個月至三個月之間的選擇權被歸類為中期選擇權，而到期期限超過三個月的選擇權則被歸類為長期選擇權。

### 3.1 買權

表 1 呈現了各模型在樣本外（out-of-sample）針對買權（call options），依據不同價內外程度（moneyness）與到期期限的 RMSE 結果。根據結果，在價外（OTM）買權方面，採用隱含波動率（implied volatility）的 NN2 模型表現最佳，其 RMSE 最小，且所有 NN 模型在各種波動率預測方法下均優於 BS 模型。在價平（ATM）選擇權中，採用十日歷史波動率的 NN2 模型為最佳模型，緊接著是採用隱含波動率的 NN1 模型，以及採用隱含波動率指數 VBI 的 NN2 模型。在價內（ITM）選擇權方面，最佳模型為採用隱含波動率的 NN2 模型，且 NN2 模型在各種波動率預測方法下均優於 BS 與 NN1 模型。另一個顯著結果是，所有模型在價內選擇權的表現均較價外與價平選擇權差，其 RMSE 幾乎大了四到五倍，這意味著價內選擇權的定價誤差高於價外與價平選擇權。就到期期限而言，採用隱含波動率的 NN2 模型在短期與中期選擇權中均為最佳模型，而採用 VBI 的 NN1 模型則在長期選擇權中表現最佳。此外，在所有到期期限維度下，大多數情況中 NN2 在各種波動率預測方法下的表現均優於 BS 與 NN1。整體結果顯示，NN2 模型在買權定價上優於 BS 與 NN1 模型。

為了呈現模型在動盪時期的表現變化，表 2 列出各模型在 2018 年 4 月交易之選擇權的 RMSE，此月份根據全樣本的 WUI 指標被視為較其他月份更為動盪的時期，並依價內外程度與到期期限分類。在價外與價平選擇權方面，採用 30 日與 21 日歷史波動率的 BS 模型表現最佳，其 RMSE 最低。在大多數波動率預測方法下，BS 的表現多數時間優於 NN1 與 NN2。在價內選擇權方面，採用 360 日歷史波動率的 NN2 模型表現最佳，緊接著是採用 30 日歷史波動率的 NN2 模型。當依到期期限評估定價表現時，短期選擇權的最佳模型為採用 360 日與 252 日歷史波動率的 BS 模型，中期選擇權的最佳模型為採用 30 日、21 日與 252 日歷史波動率的 BS 模型，而長期選擇權的最佳模型則為採用十日歷史波動率的 NN2 模型。此外，在短期選擇權上，BS 在各種波動率預測方法下均優於 NN1 與 NN2，這意味著無論使用何種波動率預測方法作為波動率輸入，在動盪時期 BS 都是短期買權定價的最佳模型。

<table>
    <caption>表 1<br />樣本外 RMSE：買權，全樣本。</caption>
    <thead>
        <tr>
            <th colspan="25">Panel A：依履約價格比率分類之模型表現</th>
        </tr>
        <tr>
            <th rowspan="2"></th>
            <th colspan="3">IMPLIED</th>
            <th colspan="3">GARCH</th>
            <th colspan="3">HIS360</th>
            <th colspan="3">HIS30</th>
            <th colspan="3">HIS10</th>
            <th colspan="3">HIS21</th>
            <th colspan="3">HIS252</th>
            <th colspan="3">VBI</th>
        </tr>
        <tr>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>OTM</td>
            <td>11.33</td>
            <td>11.45</td>
            <td>8.34</td>
            <td>26.06</td>
            <td>8.86</td>
            <td>8.77</td>
            <td>24.13</td>
            <td>12.30</td>
            <td>16.18</td>
            <td>22.05</td>
            <td>8.83</td>
            <td>14.02</td>
            <td>18.51</td>
            <td>13.99</td>
            <td>9.37</td>
            <td>18.51</td>
            <td>12.49</td>
            <td>12.20</td>
            <td>15.34</td>
            <td>9.97</td>
            <td>12.07</td>
            <td>13.35</td>
            <td>9.29</td>
            <td>9.24</td>
        </tr>
        <tr>
            <td>ATM</td>
            <td>9.78</td>
            <td>9.44</td>
            <td>10.47</td>
            <td>22.89</td>
            <td>12.76</td>
            <td>10.47</td>
            <td>21.38</td>
            <td>17.65</td>
            <td>13.52</td>
            <td>19.36</td>
            <td>11.98</td>
            <td>16.10</td>
            <td>15.59</td>
            <td>14.94</td>
            <td>9.29</td>
            <td>15.59</td>
            <td>19.03</td>
            <td>14.44</td>
            <td>14.53</td>
            <td>14.31</td>
            <td>11.31</td>
            <td>11.97</td>
            <td>9.72</td>
            <td>9.69</td>
        </tr>
        <tr>
            <td>ITM</td>
            <td>62.14</td>
            <td>58.21</td>
            <td>51.81</td>
            <td>63.89</td>
            <td>62.97</td>
            <td>57.05</td>
            <td>63.72</td>
            <td>60.89</td>
            <td>53.74</td>
            <td>63.36</td>
            <td>62.09</td>
            <td>56.91</td>
            <td>63.25</td>
            <td>60.40</td>
            <td>55.90</td>
            <td>63.25</td>
            <td>62.04</td>
            <td>57.69</td>
            <td>63.21</td>
            <td>61.40</td>
            <td>53.72</td>
            <td>62.63</td>
            <td>58.70</td>
            <td>56.19</td>
        </tr>
    </tbody>
</table>
<table>
    <thead>
        <tr>
            <th colspan="25">Panel B：依剩餘到期時間分類之模型表現</th>
        </tr>

<table>
    <thead>
        <tr>
            <th rowspan="2"></th>
            <th colspan="3">IMPLIED</th>
            <th colspan="3">GARCH</th>
            <th colspan="3">HIS360</th>
            <th colspan="3">HIS30</th>
            <th colspan="3">HIS10</th>
            <th colspan="3">HIS21</th>
            <th colspan="3">HIS252</th>
            <th colspan="3">VBI</th>
        </tr>
        <tr>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>SHORT</td>
            <td>16.58</td>
            <td>16.43</td>
            <td>16.01</td>
            <td>18.07</td>
            <td>17.98</td>
            <td>17.58</td>
            <td>18.42</td>
            <td>22.21</td>
            <td>19.55</td>
            <td>19.55</td>
            <td>20.89</td>
            <td>16.56</td>
            <td>18.42</td>
            <td>18.71</td>
            <td>20.09</td>
            <td>18.42</td>
            <td>21.49</td>
            <td>20.09</td>
            <td>16.97</td>
            <td>20.85</td>
            <td>16.71</td>
            <td>16.73</td>
            <td>16.68</td>
            <td>16.63</td>
        </tr>
        <tr>
            <td>MEDIUM</td>
            <td>16.45</td>
            <td>14.91</td>
            <td>12.16</td>
            <td>29.09</td>
            <td>15.60</td>
            <td>12.77</td>
            <td>26.95</td>
            <td>16.50</td>
            <td>13.42</td>
            <td>13.42</td>
            <td>17.06</td>
            <td>13.21</td>
            <td>21.45</td>
            <td>17.69</td>
            <td>15.67</td>
            <td>21.45</td>
            <td>18.63</td>
            <td>15.67</td>
            <td>19.14</td>
            <td>13.97</td>
            <td>13.26</td>
            <td>16.84</td>
            <td>14.23</td>
            <td>12.78</td>
        </tr>
        <tr>
            <td>LONG</td>
            <td>12.21</td>
            <td>15.15</td>
            <td>10.88</td>
            <td>46.15</td>
            <td>11.40</td>
            <td>9.43</td>
            <td>42.89</td>
            <td>14.82</td>
            <td>9.51</td>
            <td>9.51</td>
            <td>14.28</td>
            <td>9.15</td>
            <td>29.25</td>
            <td>21.96</td>
            <td>11.02</td>
            <td>29.25</td>
            <td>15.86</td>
            <td>11.02</td>
            <td>28.40</td>
            <td>11.96</td>
            <td>22.64</td>
            <td>24.84</td>
            <td>8.65</td>
            <td>13.25</td>
        </tr>
    </tbody>
</table>

Notes: NN1 與 NN2 分別代表神經網路模型，其輸入與輸出分別如方程式 (5) 與 (6) 所示。BS 為 Black-Scholes 選擇權定價模型。IMPLIED、GARCH、HIS360、HIS30、HIS10、HIS21、HIS252 與 VBI 分別代表 $\hat{\sigma}_{t}^{implied}$、$\hat{\sigma}_{t}^{garch}$、$\hat{\sigma}_{t}^{360}$、$\hat{\sigma}_{t}^{30}$、$\hat{\sigma}_{t}^{10}$、$\hat{\sigma}_{t}^{21}$、$\hat{\sigma}_{t}^{252}$ 與 $\hat{\sigma}_{t}^{vbi}$。* 表示以某特定波動率預測方法（如 ARCH）估計的最佳模型，其定價誤差與其他兩個模型的定價誤差相比，根據 Diebold-Mariano 檢定在 5% 顯著水準下具有統計顯著性。上標數字代表以某一波動率預測方法所得到的最佳模型，相對於其他波動率預測方法所得到的最佳模型的排名。例如，對於價外 (OTM) 選擇權而言，模型排名如下：1. 使用隱含波動率 (implied volatility) 的 NN2、2. 使用 GARCH 波動率的 NN2、3. 使用 30 天歷史資料估計的歷史波動率 (historical volatility) 的 NN1、4. 使用 VBI 的 NN2、5. 使用 10 天歷史資料估計的歷史波動率的 NN2、6. 使用 252 天歷史資料估計的歷史波動率的 NN1、7. 使用 21 天歷史資料估計的歷史波動率的 NN2，以及 8. 使用 360 天歷史資料估計的歷史波動率的 NN1。也就是說，對於價外選擇權而言，當使用 GARCH 波動率預測來進行選擇權定價時，最佳模型為 NN2，在不同波動率預測方法所產生的其他最佳模型中排名第二，以上標 2 表示；而當使用過去 360 天資料估計的歷史波動率預測時，最佳模型為 NN1，在其他波動率預測方法所產生的其他最佳模型中排名第八，以上標 8 表示。

Z. İltüzer

Borsa İstanbul Review 22-4 (2022) 725–742

<table>
<thead>
<tr>
<th></th>
<th colspan="2">IMPLIED</th>
<th colspan="2">GARCH</th>
<th colspan="2">HIS360</th>
<th colspan="2">HIS30</th>
<th colspan="2">HIS10</th>
<th colspan="2">HIS21</th>
<th colspan="2">HIS252</th>
<th colspan="2">VBI</th>
</tr>
<tr>
<th></th>
<th>BS</th>
<th>NN1</th>
<th>NN2</th>
<th>BS</th>
<th>NN1</th>
<th>NN2</th>
<th>BS</th>
<th>NN1</th>
<th>NN2</th>
<th>BS</th>
<th>NN1</th>
<th>NN2</th>
<th>BS</th>
<th>NN1</th>
<th>NN2</th>
<th>BS</th>
<th>NN1</th>
<th>NN2</th>
</tr>
<tr>
<td>OTM</td>
<td>0.67</td>
<td>0.65</td>
<td>0.58</td>
<td>0.56</td>
<td>0.45</td>
<td>0.64</td>
<td>0.55</td>
<td>0.79</td>
<td>0.52</td>
<td>0.49</td>
<td>0.36</td>
<td>0.41</td>
<td>0.37</td>
<td>0.41</td>
<td>0.93</td>
<td>0.72</td>
<td>0.72</td>
</tr>
<tr>
<td>ATM</td>
<td>0.84</td>
<td>0.94</td>
<td>0.85</td>
<td>0.84</td>
<td>0.77</td>
<td>1.12</td>
<td>1.05</td>
<td>0.65</td>
<td>0.73</td>
<td>1.04</td>
<td>0.65</td>
<td>0.81</td>
<td>0.63</td>
<td>0.81</td>
<td>1.86</td>
<td>1.02</td>
<td>0.87</td>
</tr>
<tr>
<td>ITM</td>
<td>2.02</td>
<td>2.34</td>
<td>2.10</td>
<td>2.02</td>
<td>2.01</td>
<td>1.84</td>
<td>1.58</td>
<td>1.92</td>
<td>2.05</td>
<td>1.85</td>
<td>2.03</td>
<td>1.82</td>
<td>2.00</td>
<td>1.82</td>
<td>2.58</td>
<td>2.02</td>
<td>1.99</td>
</tr>
</thead>
<tbody>
<tr>
<td colspan="18">Panel B: 依到期時間分類之模型績效</td>
</tr>
<tr>
<td>SHORT</td>
<td>0.55</td>
<td>0.69</td>
<td>0.77</td>
<td>0.55</td>
<td>0.85</td>
<td>0.54</td>
<td>0.93</td>
<td>0.74</td>
<td>0.65</td>
<td>0.83</td>
<td>0.57</td>
<td>0.59</td>
<td>0.54</td>
<td>0.59</td>
<td>1.39</td>
<td>0.61</td>
<td>0.77</td>
</tr>
<tr>
<td>MEDIUM</td>
<td>1.11</td>
<td>1.05</td>
<td>0.83</td>
<td>1.05</td>
<td>1.19</td>
<td>0.91</td>
<td>0.62</td>
<td>0.73</td>
<td>1.65</td>
<td>1.01</td>
<td>0.69</td>
<td>0.80</td>
<td>0.69</td>
<td>0.80</td>
<td>1.50</td>
<td>1.28</td>
<td>0.71</td>
</tr>
<tr>
<td>LONG</td>
<td>2.02</td>
<td>3.23</td>
<td>2.79</td>
<td>0.51</td>
<td>1.78</td>
<td>0.42</td>
<td>3.07</td>
<td>1.21</td>
<td>2.83</td>
<td>1.48</td>
<td>0.19</td>
<td>1.72</td>
<td>1.45</td>
<td>1.72</td>
<td>4.00</td>
<td>2.40</td>
<td>5.15</td>
</tr>
<tr>
<td colspan="18"><b>註：</b> NN1 與 NN2 分別代表其輸入與輸出如方程式 <sup>(5)</sup> 與 <sup>(6)</sup> 所述的神經網路模型。BS 為 Black-Scholes 選擇權定價模型。IMPLIED、GARCH、HIS360、HIS30、HIS10、HIS21、HIS252 與 VBI 分別代表 $\hat{\sigma}^{implied}_{t}$、$\hat{\sigma}^{garch}_{t}$、$\hat{\sigma}^{360}_{t}$、$\hat{\sigma}^{30}_{t}$、$\hat{\sigma}^{10}_{t}$、$\hat{\sigma}^{21}_{t}$、$\hat{\sigma}^{252}_{t}$ 以及 VBI。* 表示以特定波動率預測方法（如 ARCH）估計的最佳模型，其定價誤差相對於其他兩個模型的定價誤差，根據 Diebold-Mariano 檢定在 5% 顯著水準下具有統計顯著性。例如，對於 OTM 選擇權而言，模型排名如下：1. 使用 30 天歷史波動率預測方法的 BS 模型，相較於其他波動率預測方法的最佳模型。3. 使用 252 天歷史波動率的 BS。4. 使用 360 天歷史波動率的 BS。5. 使用 10 天歷史波動率的 NN2。6. 使用 GARCH 波動率的 BS。7. 使用隱含波動率的 NN2。8. 使用 VBI 的 BS。也就是說，對於 OTM 選擇權而言，當使用 21 天歷史波動率預測來為選擇權定價時，最佳模型為 BS，且在不同波動率預測下，它在其他最佳模型中排名第二，以上標 2 表示，而在其他波動率預測下，BS 則為最佳模型。</td>
</tr>
</tbody>
</table>

Figs. 2 與 3 分別描繪了各模型在 moneyness 維度下，全樣本與子樣本的定價誤差（$c - \hat{c}$）；而<u>Figs. 4 和 5</u> 則分別顯示各模型在剩餘到期時間維度下，全樣本與子樣本的定價誤差。

這些圖形呈現了在採用不同波動率預測方法時，各模型在不同 moneyness 水準及不同剩餘到期時間下的表現。大多數模型都存在定價偏差，不是高估就是低估買權。使用隱含波動率（implied volatility）的 BS 模型會低估價平（ATM）與價外（OTM）選擇權，而使用隱含波動率的 NN2 則會高估價平選擇權。使用 GARCH 波動率的 BS 與 NN1、使用過去 360 天資料計算的歷史波動率的 BS 與 NN1、使用過去 30 天與 10 天資料估計的歷史波動率的 BS、使用最近 10 天資料估計的歷史波動率的 NN1，在大多數情況下都會低估價外、價平與價內（ITM）買權。

相對地，使用過去 30 天與 360 天資料估計的歷史波動率的 NN2 會高估價外與價平買權；使用過去 21 天與 252 個交易日資料估計的歷史波動率的 BS 傾向高估價平與價外選擇權；使用過去 21 個交易日資料估計的歷史波動率以及使用 VBI 的 NN2 傾向高估價外、價平與價內買權；而使用 VBI 的 NN1 則傾向高估價外選擇權。簡而言之，BS 模型在幾乎所有波動率輸入下都偏向低估價平與價外買權，而 NN1 與 NN2 在不同波動率估計下並未呈現這種一致性的低估或高估偏差。NN1 與 NN2 模型在使用某些波動率輸入時傾向高估買權，但在使用其他波動率輸入時則傾向低估。<u>Table 1</u> 與 <u>Fig. 2</u> 的結果共同顯示，使用過去 10 天資料估計的歷史波動率的 NN2 是價平買權的最佳模型，但該模型傾向高估選擇權。因此，實務工作者必須考量此高估行為對其投資或避險策略的影響。

由於土耳其是新興市場與開發中國家，其衍生性金融商品市場與生態系仍處於發展初期，這可由低交易量、普遍使用 BS 模型定價，以及大多數市場參與者缺乏運用複雜先進選擇權定價模型的教育背景可見一斑。因此，BS 模型系統性的低估現象，可能來自於使用該模型的市場參與者在下單時，會在計算出的價格上加上模型風險溢酬（model risk premium）。在子樣本中，依 moneyness 維度分析的<u>Fig. 3</u> 也呈現出與全樣本分析相同的 BS 模型低估型態。然而，在全樣本分析中未觀察到的 NN1 與 NN2 模型在使用各種波動率預測方法時，對價平及／或價外選擇權的一致性高估行為，卻出現在子樣本中，這意味著 NN 選擇權定價模型在市場動盪時期傾向高估買權。

<u>Figs. 4 和 5</u> 呈現了各模型在全樣本中依剩餘到期時間維度所計算的定價誤差。

730

<table>
  <caption>BS、NN1與NN2模型在不同波動率預測方法下的定價誤差 — x軸：定價誤差；y軸：價內程度（ITM、ATM、OTM）</caption>
  <thead>
    <tr>
      <th rowspan="2">波動率預測方法</th>
      <th colspan="3">模型</th>
    </tr>
    <tr>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$\hat{\sigma}_t^{implied}$</th>
      <td>(a) BS($\hat{\sigma}_t^{implied}$)</td>
      <td>(b) NN1($\hat{\sigma}_t^{implied}$)</td>
      <td>(c) NN2($\hat{\sigma}_t^{implied}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{garch}$</th>
      <td>(d) BS($\hat{\sigma}_t^{garch}$)</td>
      <td>(e) NN1($\hat{\sigma}_t^{garch}$)</td>
      <td>(f) NN2($\hat{\sigma}_t^{garch}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{360}$</th>
      <td>(g) BS($\hat{\sigma}_t^{360}$)</td>
      <td>(h) NN1($\hat{\sigma}_t^{360}$)</td>
      <td>(i) NN2($\hat{\sigma}_t^{360}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{30}$</th>
      <td>(j) BS($\hat{\sigma}_t^{30}$)</td>
      <td>(k) NN1($\hat{\sigma}_t^{30}$)</td>
      <td>(l) NN2($\hat{\sigma}_t^{30}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{10}$</th>
      <td>(m) BS($\hat{\sigma}_t^{10}$)</td>
      <td>(n) NN1($\hat{\sigma}_t^{10}$)</td>
      <td>(o) NN2($\hat{\sigma}_t^{10}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{21}$</th>
      <td>(p) BS($\hat{\sigma}_t^{21}$)</td>
      <td>(q) NN1($\hat{\sigma}_t^{21}$)</td>
      <td>(r) NN2($\hat{\sigma}_t^{21}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{252}$</th>
      <td>(s) BS($\hat{\sigma}_t^{252}$)</td>
      <td>(t) NN1($\hat{\sigma}_t^{252}$)</td>
      <td>(u) NN2($\hat{\sigma}_t^{252}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{vbi}$</th>
      <td>(v) BS($\hat{\sigma}_t^{vbi}$)</td>
      <td>(w) NN1($\hat{\sigma}_t^{vbi}$)</td>
      <td>(x) NN2($\hat{\sigma}_t^{vbi}$)</td>
    </tr>
  </tbody>
</table>

圖 2. 依價內程度區分的各模型定價誤差（買權）：全樣本。

<table>
  <caption>BIST 30 指數選擇權依模型與波動率預測方法之定價誤差——面板 (a) 至 (x)</caption>
  <thead>
    <tr>
      <th rowspan="2">波動率預測方法</th>
      <th rowspan="2">價內外程度</th>
      <th colspan="3">依模型之定價誤差 (x 軸)</th>
    </tr>
    <tr>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3">(a, b, c) $\hat{\sigma}_t^{implied}$</th>
      <th>ITM</th>
      <td>-2.0, -1.5, 0.0, 0.5, 4.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.5</td>
      <td>-2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th rowspan="3">(d, e, f) $\hat{\sigma}_t^{garch}$</th>
      <th>ITM</th>
      <td>-1.0, 0.0, 1.0, 4.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-0.5, 0.0, 0.5, 2.0, 4.0</td>
      <td>-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-0.5, 0.0, 0.5, 1.5, 3.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th rowspan="3">(g, h, i) $\hat{\sigma}_t^{360}$</th>
      <th>ITM</th>
      <td>-1.0, 0.0, 1.0, 4.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 2.5</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 3.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-0.5, 0.0, 0.5, 2.0, 4.0</td>
      <td>-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5</td>
      <td>-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-0.5, 0.0, 0.5, 1.5, 3.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0</td>
    </tr>
    <tr>
      <th rowspan="3">(j, k, l) $\hat{\sigma}_t^{30}$</th>
      <th>ITM</th>
      <td>-2.5, -2.0, 0.0, 1.0, 4.5</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 2.5</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 3.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.5</td>
      <td>-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0</td>
    </tr>
    <tr>
      <th rowspan="3">(m, n, o) $\hat{\sigma}_t^{10}$</th>
      <th>ITM</th>
      <td>-1.0, 0.0, 1.0, 4.0</td>
      <td>-4.0, -1.0, 0.0, 1.0, 2.5</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 3.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-0.5, 0.0, 0.5, 2.0, 4.0</td>
      <td>-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-0.5, 0.0, 0.5, 1.5, 3.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0</td>
    </tr>
    <tr>
      <th rowspan="3">(p, q, r) $\hat{\sigma}_t^{21}$</th>
</table>

<table>
  <tbody>
    <tr>
      <th>ITM</th>
      <td>-1.0, 0.0, 1.0, 4.0, 6.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 2.5</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 3.0, 5.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-0.5, 0.0, 0.5, 2.0, 4.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0</td>
      <td>-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-0.5, 0.0, 0.5, 1.5, 3.0</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.5</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5</td>
    </tr>
    <tr>
      <th rowspan="3">(s, t, u) $\hat{\sigma}_t^{252}$</th>
      <th>ITM</th>
      <td>-1.0, 0.0, 1.0, 4.0</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 2.5</td>
      <td>-2.0, -1.5, 0.0, 1.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-0.5, 0.0, 0.5, 2.0, 4.0</td>
      <td>-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-0.5, 0.0, 0.5, 1.5, 3.0</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0</td>
    </tr>
    <tr>
      <th rowspan="3">(v, w, x) $\hat{\sigma}_t^{vbi}$</th>
      <th>ITM</th>
      <td>-1.0, 0.0, 1.0, 4.0</td>
      <td>-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.5</td>
      <td>-3.5, -2.0, -1.0, 0.0, 1.0, 3.0</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-0.5, 0.0, 0.5, 2.0, 4.0</td>
      <td>-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
      <td>-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-0.5, 0.0, 0.5, 1.5, 3.0</td>
      <td>-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5</td>
      <td>-1.0, -0.5, 0.0, 0.5, 1.0, 2.0</td>
    </tr>
  </tbody>
</table>

圖 3. 依據價內程度之模型定價誤差（買權）：子樣本。

Z. İltüzer

Borsa İstanbul Review 22-4 (2022) 725–742

<table>
<thead>
<tr>
<th>標籤</th>
<th>模型</th>
<th>x軸範圍</th>
<th>y軸範圍</th>
</tr>
</thead>
<tbody>
<tr>
<td>(a)</td>
<td>BS($\hat{\sigma}_t^{implied}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(b)</td>
<td>NN1 ($\hat{\sigma}_t^{implied}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(c)</td>
<td>NN2 ($\hat{\sigma}_t^{implied}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(d)</td>
<td>BS($\hat{\sigma}_t^{garch}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(e)</td>
<td>NN1 ($\hat{\sigma}_t^{garch}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(f)</td>
<td>NN2 ($\hat{\sigma}_t^{garch}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(g)</td>
<td>BS($\hat{\sigma}_t^{360}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(h)</td>
<td>NN1 ($\hat{\sigma}_t^{360}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(i)</td>
<td>NN2 ($\hat{\sigma}_t^{360}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(j)</td>
<td>BS($\hat{\sigma}_t^{30}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(k)</td>
<td>NN1 ($\hat{\sigma}_t^{30}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(l)</td>
<td>NN2 ($\hat{\sigma}_t^{30}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(m)</td>
<td>BS($\hat{\sigma}_t^{10}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(n)</td>
<td>NN1 ($\hat{\sigma}_t^{10}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(o)</td>
<td>NN2 ($\hat{\sigma}_t^{10}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(p)</td>
<td>BS($\hat{\sigma}_t^{21}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(q)</td>
<td>NN1 ($\hat{\sigma}_t^{21}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(r)</td>
<td>NN2 ($\hat{\sigma}_t^{21}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(s)</td>
<td>BS($\hat{\sigma}_t^{252}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(t)</td>
<td>NN1 ($\hat{\sigma}_t^{252}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(u)</td>
<td>NN2 ($\hat{\sigma}_t^{252}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(v)</td>
<td>BS($\hat{\sigma}_t^{pbi}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(w)</td>
<td>NN1 ($\hat{\sigma}_t^{pbi}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
<tr>
<td>(x)</td>
<td>NN2 ($\hat{\sigma}_t^{pbi}$)</td>
<td>-100 to 100</td>
<td>-100 to 140</td>
</tr>
</tbody>
</table>

圖 4. 各模型依到期時間之定價誤差（買權）：全樣本。

733

<table>
  <caption>依據價內程度與波動率預測方法之定價誤差——面板 (a) 至 (x)</caption>
  <thead>
    <tr>
      <th rowspan="2">波動率預測方法</th>
      <th colspan="3">模型</th>
    </tr>
    <tr>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$\hat{\sigma}_t^{implied}$</th>
      <td>(a) BS($\hat{\sigma}_t^{implied}$)</td>
      <td>(b) NN1($\hat{\sigma}_t^{implied}$)</td>
      <td>(c) NN2($\hat{\sigma}_t^{implied}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{garch}$</th>
      <td>(d) BS($\hat{\sigma}_t^{garch}$)</td>
      <td>(e) NN1($\hat{\sigma}_t^{garch}$)</td>
      <td>(f) NN2($\hat{\sigma}_t^{garch}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{360}$</th>
      <td>(g) BS($\hat{\sigma}_t^{360}$)</td>
      <td>(h) NN1($\hat{\sigma}_t^{360}$)</td>
      <td>(i) NN2($\hat{\sigma}_t^{360}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{30}$</th>
      <td>(j) BS($\hat{\sigma}_t^{30}$)</td>
      <td>(k) NN1($\hat{\sigma}_t^{30}$)</td>
      <td>(l) NN2($\hat{\sigma}_t^{30}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{10}$</th>
      <td>(m) BS($\hat{\sigma}_t^{10}$)</td>
      <td>(n) NN1($\hat{\sigma}_t^{10}$)</td>
      <td>(o) NN2($\hat{\sigma}_t^{10}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{21}$</th>
      <td>(p) BS($\hat{\sigma}_t^{21}$)</td>
      <td>(q) NN1($\hat{\sigma}_t^{21}$)</td>
      <td>(r) NN2($\hat{\sigma}_t^{21}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{252}$</th>
      <td>(s) BS($\hat{\sigma}_t^{252}$)</td>
      <td>(t) NN1($\hat{\sigma}_t^{252}$)</td>
      <td>(u) NN2($\hat{\sigma}_t^{252}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{vbi}$</th>
      <td>(v) BS($\hat{\sigma}_t^{vbi}$)</td>
      <td>(w) NN1($\hat{\sigma}_t^{vbi}$)</td>
      <td>(x) NN2($\hat{\sigma}_t^{vbi}$)</td>
    </tr>
  </tbody>
</table>
<table>
  <caption>依價內程度分類之定價誤差分布——y軸：定價誤差；x軸：剩餘到期時間</caption>
  <thead>
    <tr>
      <th>價內程度（y軸標籤）</th>
      <th>定價誤差值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ITM</th>
      <td>140</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>90</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>30</td>
    </tr>
  </tbody>
</table>

圖 5. 各模型依剩餘到期時間之定價誤差（買權）：子樣本。

**Table 3**  
樣本外 RMSE 值－賣權（put options）－完整樣本。

<table>
    <thead>
        <tr>
            <th colspan="25">Panel A: 依履約價格分類之模型績效</th>
        </tr>
        <tr>
            <th></th>
            <th colspan="3">IMPLIED</th>
            <th colspan="3">GARCH</th>
            <th colspan="3">HIS360</th>
            <th colspan="3">HIS30</th>
            <th colspan="3">HIS10</th>
            <th colspan="3">HIS21</th>
            <th colspan="3">HIS252</th>
            <th colspan="3">VBI</th>
        </tr>
        <tr>
            <th></th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
            <th>BS</th>
            <th>NN1</th>
            <th>NN2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>OTM</td>
            <td>27.01<sup>7</sup></td>
            <td>26.31</td>
            <td>23.86</td>
            <td>24.66<sup>8</sup></td>
            <td>53.88</td>
            <td>38.29</td>
            <td>13.34<sup>1</sup>*</td>
            <td>36.42</td>
            <td>19.29</td>
            <td>23.67<sup>4</sup></td>
            <td>48.45</td>
            <td>18.46</td>
            <td>18.78<sup>6</sup></td>
            <td>29.44</td>
            <td>21.56</td>
            <td>18.78<sup>6</sup></td>
            <td>36.95</td>
            <td>23.55</td>
            <td>18.31<sup>3</sup></td>
            <td>35.44</td>
            <td>20.05</td>
            <td>16.52<sup>2</sup></td>
            <td>24.73</td>
            <td>17.28</td>
        </tr>
        <tr>
            <td>ATM</td>
            <td>23.20</td>
            <td>14.79<sup>6</sup></td>
            <td>14.64</td>
            <td>15.45</td>
            <td>20.59</td>
            <td>16.24<sup>8</sup></td>
            <td>8.89<sup>1</sup>*</td>
            <td>14.66</td>
            <td>11.10</td>
            <td>20.16</td>
            <td>27.25</td>
            <td>13.92<sup>5</sup></td>
            <td>16.14</td>
            <td>20.03</td>
            <td>12.29<sup>3</sup></td>
            <td>16.14</td>
            <td>23.44</td>
            <td>14.99<sup>7</sup></td>
            <td>13.70<sup>2</sup></td>
            <td>18.89</td>
            <td>11.93</td>
            <td>13.97</td>
            <td>16.13</td>
            <td>13.42<sup>4</sup></td>
        </tr>
        <tr>
            <td>ITM</td>
            <td>19.86</td>
            <td>10.19<sup>4</sup></td>
            <td>13.40</td>
            <td>17.90</td>
            <td>14.81</td>
            <td>14.73<sup>8</sup></td>
            <td>7.69<sup>1</sup>*</td>
            <td>8.84</td>
            <td>8.87</td>
            <td>17.85</td>
            <td>12.66</td>
            <td>9.67<sup>5</sup></td>
            <td>13.60</td>
            <td>13.45</td>
            <td>12.16<sup>7</sup></td>
            <td>13.60</td>
            <td>12.00<sup>6</sup></td>
            <td>15.03</td>
            <td>14.11</td>
            <td>12.44</td>
            <td>10.66<sup>3</sup></td>
            <td>13.00</td>
            <td>12.19</td>
        </tr>
    </tbody>
</table>

<table>
        <tr>
            <td>8.87<sup>2</sup></td>
        </tr>
        <tr>
            <th colspan="25">Panel B: 依到期時間分類之模型績效</th>
        </tr>
        <tr>
            <td>SHORT</td>
            <td>10.94<sup>5</sup></td>
            <td>11.56</td>
            <td>15.31</td>
            <td>7.11<sup>2</sup></td>
            <td>13.38</td>
            <td>15.60</td>
            <td>7.13<sup>3</sup></td>
            <td>12.28</td>
            <td>14.20</td>
            <td>12.56</td>
            <td>20.70</td>
            <td>12.24<sup>7</sup></td>
            <td>8.51<sup>4</sup></td>
            <td>11.72</td>
            <td>11.92</td>
            <td>8.51<sup>4</sup></td>
            <td>14.61</td>
            <td>13.56</td>
            <td>7.96<sup>6</sup></td>
            <td>9.36</td>
            <td>13.06</td>
            <td>6.85<sup>1</sup>*</td>
            <td>14.24</td>
            <td>10.38</td>
        </tr>
        <tr>
            <td>MEDIUM</td>
            <td>25.29</td>
            <td>16.91</td>
            <td>15.82<sup>7</sup></td>
            <td>20.35</td>
            <td>35.20</td>
            <td>25.24<sup>8</sup></td>
            <td>9.98<sup>1</sup>*</td>
            <td>16.90</td>
            <td>11.27</td>
            <td>21.51</td>
            <td>29.57</td>
            <td>14.02<sup>4</sup></td>
            <td>16.98</td>
            <td>18.73</td>
            <td>14.96<sup>6</sup></td>
            <td>16.98</td>
            <td>24.21</td>
            <td>17.41<sup>5</sup></td>
            <td>15.47</td>
            <td>20.40</td>
            <td>13.04<sup>2</sup></td>
            <td>15.31</td>
            <td>15.41</td>
            <td>14.03<sup>3</sup></td>
        </tr>
        <tr>
            <td>LONG</td>
            <td>42.28</td>
            <td>29.42</td>
            <td>23.92<sup>6</sup></td>
            <td>33.70</td>
            <td>38.55</td>
            <td>25.18<sup>7</sup></td>
            <td>15.41</td>
            <td>45.47</td>
            <td>11.92<sup>1</sup>*</td>
            <td>35.36</td>
            <td>54.66</td>
            <td>19.24<sup>5</sup></td>
            <td>30.05</td>
            <td>46.77</td>
            <td>20.96<sup>8</sup></td>
            <td>30.05</td>
            <td>48.06</td>
            <td>25.95<sup>4</sup></td>
            <td>28.40</td>
            <td>50.30</td>
            <td>18.00<sup>2</sup></td>
            <td>26.82</td>
            <td>32.49</td>
            <td>18.36<sup>3</sup></td>
        </tr>
    </tbody>
</table>
註：NN1與NN2分別代表其輸入與輸出如方程式(5)與(6)所述的神經網路模型。BS為Black-Scholes選擇權定價模型。IMPLIED、GARCH、HIS360、HIS30、HIS10、HIS21、HIS252與VBI分別代表σ̂<sub>t</sub><sup>implied</sup>、σ̂<sub>t</sub><sup>garch</sup>、σ̂<sub>t</sub><sup>360</sup>、σ̂<sub>t</sub><sup>30</sup>、σ̂<sub>t</sub><sup>10</sup>、σ̂<sub>t</sub><sup>21</sup>、σ̂<sub>t</sub><sup>252</sup>與σ̂<sub>t</sub><sup>vbi</sup>。*表示以特定波動率預測方法（如ARCH）估計之最佳模型，其定價誤差根據Diebold-Mariano檢定，在5%顯著水準下，相較於其他兩種模型的定價誤差具有統計顯著性。上標數字代表在某一波動率預測方法下之最佳模型，相對於其他波動率預測方法下之最佳模型的排名。

BS 模型在不同波動率預測方法下的表現。例如，對於價外（OTM）選擇權而言，各模型的排名如下：1. 使用 360 天歷史波動率（historical volatility）的 BS 模型，2. 使用 VBI 的 BS 模型，3. 使用 252 天歷史波動率的 BS 模型，4. 使用 30 天歷史波動率的 BS 模型，5. 使用 10 天歷史波動率的 NN2 模型，6. 使用 21 天歷史波動率的 BS 模型，7. 使用隱含波動率（implied volatility）的 BS 模型，以及 8. 使用 GARCH 的 BS 模型。也就是說，對於價外選擇權而言，當使用 VBI 進行選擇權定價時，BS 模型是最佳模型，而當使用其他不同波動率預測時，它在其他最佳模型中排名第二（以右上角數字 2 表示）；同時，當使用過去 30 天資料的歷史波動率預測時，BS 模型是最佳模型，而在其他波動率預測下的最佳模型中，其排名為第四（以右上角數字 4 表示）。

735

表 4  
樣本外賣權模型 RMSE 值－子樣本

<table>
  <thead>
    <tr>
      <th colspan="25">Panel A：依價內外程度分類之模型績效</th>
    </tr>
    <tr>
      <th></th>
      <th colspan="3">IMPLIED</th>
      <th colspan="3">GARCH</th>
      <th colspan="3">HIS360</th>
      <th colspan="3">HIS30</th>
      <th colspan="3">HIS10</th>
      <th colspan="3">HIS21</th>
      <th colspan="3">HIS252</th>
      <th colspan="3">VBI</th>
    </tr>
    <tr>
      <th></th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OTM</td>
      <td>1.22<sup>3</sup></td>
      <td>1.22</td>
      <td>2.36</td>
      <td>1.20<sup>2</sup></td>
      <td>1.27</td>
      <td>2.36</td>
      <td>1.25<sup>4</sup></td>
      <td>1.37</td>
      <td>2.59</td>
      <td>1.55<sup>8</sup></td>
      <td>1.65</td>
      <td>2.54</td>
      <td>1.51<sup>5</sup></td>
      <td>1.30</td>
      <td>2.54</td>
      <td>1.55<sup>7</sup></td>
      <td>1.78</td>
      <td>3.02</td>
      <td>1.63</td>
      <td>1.47<sup>6</sup></td>
      <td>2.74</td>
      <td>1.04<sup>1</sup></td>
      <td>1.30</td>
      <td>1.80</td>
    </tr>
    <tr>
      <td>ATM</td>
      <td>0.60<sup>1</sup></td>
      <td>0.60</td>
      <td>0.72</td>
      <td>0.84</td>
      <td>0.62<sup>3</sup></td>
      <td>0.66</td>
      <td>0.93</td>
      <td>0.59<sup>2</sup></td>
      <td>0.63</td>
      <td>1.13</td>
      <td>0.91</td>
      <td>0.79<sup>8</sup></td>
      <td>1.09</td>
      <td>0.73<sup>7</sup></td>
      <td>0.76</td>
      <td>1.13</td>
      <td>0.90</td>
      <td>0.92</td>
      <td>1.31</td>
      <td>0.83</td>
      <td>0.68<sup>6</sup></td>
      <td>0.71</td>
      <td>0.63<sup>4</sup></td>
      <td>0.67<sup>5</sup></td>
    </tr>
    <tr>
      <td>ITM</td>
      <td>0.59<sup>5</sup></td>
      <td>0.59</td>
      <td>0.54<sup>4</sup></td>
      <td>1.12</td>
      <td>0.77</td>
      <td>0.55<sup>6</sup></td>
      <td>1.19</td>
      <td>0.45</td>
      <td>0.37<sup>1</sup></td>
      <td>1.52</td>
      <td>1.01</td>
      <td>0.48<sup>3</sup></td>
      <td>1.24</td>
      <td>0.70</td>
      <td>0.49</td>
      <td>1.52</td>
      <td>1.18</td>
      <td>0.73</td>
      <td>1.52</td>
      <td>0.84</td>
      <td>0.42<sup>2</sup></td>
      <td>0.98</td>
      <td>0.70</td>
      <td>0.58<sup>7</sup></td>
    </tr>
    <tr>
      <th colspan="25">Panel B：依剩餘到期時間分類之模型績效</th>
    </tr>
    <tr>
      <td>SHORT</td>
      <td>0.87</td>
      <td>0.65<sup>3</sup></td>
      <td>1.35</td>
      <td>0.71<sup>5</sup></td>
      <td>0.73</td>
      <td>1.33</td>
      <td>0.70<sup>4</sup></td>
      <td>0.71</td>
      <td>1.30</td>
      <td>0.93</td>
      <td>1.02</td>
      <td>1.30</td>
      <td>0.90</td>
      <td>0.78<sup>7</sup></td>
      <td>1.45</td>
      <td>0.93</td>
      <td>1.09</td>
      <td>1.48</td>
      <td>0.94</td>
      <td>0.96<sup>8</sup></td>
      <td>1.51</td>
      <td>0.60<sup>1</sup></td>
      <td>0.78<sup>6</sup></td>
      <td>1.03<sup>2</sup></td>
    </tr>
    <tr>
      <td>MED</td>
</table>

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="3">NN1</th>
      <th colspan="3">NN2</th>
      <th colspan="3">BS</th>
    </tr>
    <tr>
      <th> moneyness </th>
      <th>IMPLIED</th>
      <th>GARCH</th>
      <th>HIS360</th>
      <th>HIS30</th>
      <th>HIS10</th>
      <th>HIS21</th>
      <th>HIS252</th>
      <th>VBI</th>
      <th>IMPLIED</th>
      <th>GARCH</th>
      <th>HIS360</th>
      <th>HIS30</th>
      <th>HIS10</th>
      <th>HIS21</th>
      <th>HIS252</th>
      <th>VBI</th>
      <th>IMPLIED</th>
      <th>GARCH</th>
      <th>HIS360</th>
      <th>HIS30</th>
      <th>HIS10</th>
      <th>HIS21</th>
      <th>HIS252</th>
      <th>VBI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OTM</td>
      <td>1.69</td>
      <td>0.86<sup>6</sup></td>
      <td>0.76<sup>3</sup></td>
      <td>1.25</td>
      <td>0.88</td>
      <td>0.71<sup>1</sup></td>
      <td>1.40</td>
      <td>0.82<sup>4</sup></td>
      <td>0.97</td>
      <td>1.69</td>
      <td>1.17</td>
      <td>1.12</td>
      <td>1.54</td>
      <td>0.93<sup>8</sup></td>
      <td>0.77<sup>5</sup></td>
      <td>1.69</td>
      <td>1.23</td>
      <td>1.45</td>
      <td>1.90</td>
      <td>0.98</td>
      <td>0.72<sup>2</sup></td>
      <td>1.08</td>
      <td>0.80<sup>7</sup></td>
      <td>0.79</td>
    </tr>
    <tr>
      <td>ATM</td>
      <td>1.69</td>
      <td>0.86<sup>6</sup></td>
      <td>0.76<sup>3</sup></td>
      <td>1.25</td>
      <td>0.88</td>
      <td>0.71<sup>1</sup></td>
      <td>1.40</td>
      <td>0.82<sup>4</sup></td>
      <td>0.97</td>
      <td>1.69</td>
      <td>1.17</td>
      <td>1.12</td>
      <td>1.54</td>
      <td>0.93<sup>8</sup></td>
      <td>0.77<sup>5</sup></td>
      <td>1.69</td>
      <td>1.23</td>
      <td>1.45</td>
      <td>1.90</td>
      <td>0.98</td>
      <td>0.72<sup>2</sup></td>
      <td>1.08</td>
      <td>0.80<sup>7</sup></td>
      <td>0.79</td>
    </tr>
    <tr>
      <td>ITM</td>
      <td>1.69</td>
      <td>0.86<sup>6</sup></td>
      <td>0.76<sup>3</sup></td>
      <td>1.25</td>
      <td>0.88</td>
      <td>0.71<sup>1</sup></td>
      <td>1.40</td>
      <td>0.82<sup>4</sup></td>
      <td>0.97</td>
      <td>1.69</td>
      <td>1.17</td>
      <td>1.12</td>
      <td>1.54</td>
      <td>0.93<sup>8</sup></td>
      <td>0.77<sup>5</sup></td>
      <td>1.69</td>
      <td>1.23</td>
      <td>1.45</td>
      <td>1.90</td>
      <td>0.98</td>
      <td>0.72<sup>2</sup></td>
      <td>1.08</td>
      <td>0.80<sup>7</sup></td>
      <td>0.79</td>
    </tr>
    <tr>
      <td>LONG</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
      <td>–</td>
    </tr>
  </tbody>
</table>

註：NN1 與 NN2 分別代表其輸入與輸出如方程式 (5) 與 (6) 所示的神經網路模型。BS 為 Black-Scholes 選擇權定價模型。IMPLIED、GARCH、HIS360、HIS30、HIS10、HIS21、HIS252 與 VBI 分別代表 σ̂<sub>t</sub><sup>implied</sup>、σ̂<sub>t</sub><sup>garch</sup>、σ̂<sub>t</sub><sup>360</sup>、σ̂<sub>t</sub><sup>30</sup>、σ̂<sub>t</sub><sup>10</sup>、σ̂<sub>t</sub><sup>21</sup>、σ̂<sub>t</sub><sup>252</sup> 與 σ̂<sub>t</sub><sup>vbi</sup>。* 表示以某種波動率預測方法（如 ARCH）估計的最佳模型，其定價誤差相對於其他兩種模型的定價誤差，根據 Diebold-Mariano 檢定在 5% 顯著水準下具有統計顯著性。上標數字代表在某一波動率預測方法下最佳模型，相對於其他波動率預測方法下最佳模型的排名。例如，在價外 (OTM) 選擇權中，各模型的排名如下：1. 使用 VBI 的 BS、2. 使用 GARCH 的 BS、3. 使用隱含波動率 (implied volatility) 的 BS、4. 使用 360 天歷史波動率的 BS、5. 使用 10 天歷史波動率的 BS、6. 使用 252 天歷史波動率的 NN1、7. 使用 21 天歷史波動率的 BS、8. 使用 30 天歷史波動率的 BS。也就是說，在價外選擇權中，當使用 GARCH 波動率預測來定價時，最佳模型為 BS，且在不同波動率預測方法下其他最佳模型中排名第二（以數字 2 表示）；而當使用過去 360 天資料的歷史波動率預測時，BS 為最佳模型，在其他波動率預測方法下的最佳模型中排名第四（以數字 4 表示）。

<table>
  <caption>不同模型與波動率預測方法在各價內程度類別的定價誤差——面板 (a) 至 (x)</caption>
  <thead>
    <tr>
      <th rowspan="2">波動率預測方法</th>
      <th rowspan="2">價內程度 (Moneyness)</th>
      <th colspan="3">模型定價誤差範圍（約略）</th>
    </tr>
    <tr>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3">隱含波動率 ($\hat{\sigma}_t^{implied}$)</th>
      <th>價內 (ITM)</th>
      <td>(a) -25 to 50</td>
      <td>(b) -25 to 25</td>
      <td>(c) -25 to 50</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(a) -50 to 50</td>
      <td>(b) -25 to 25</td>
      <td>(c) -40 to 50</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(a) -25 to 50</td>
      <td>(b) -75 to 25</td>
      <td>(c) -50 to 50</td>
    </tr>
    <tr>
      <th rowspan="3">GARCH ($\hat{\sigma}_t^{garch}$)</th>
      <th>價內 (ITM)</th>
      <td>(d) -50 to 75</td>
      <td>(e) -50 to 50</td>
      <td>(f) -50 to 50</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(d) -50 to 100</td>
      <td>(e) -75 to 75</td>
      <td>(f) -75 to 75</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(d) -50 to 100</td>
      <td>(e) -100 to 50</td>
      <td>(f) -100 to 100</td>
    </tr>
    <tr>
      <th rowspan="3">360天 ($\hat{\sigma}_t^{360}$)</th>
      <th>價內 (ITM)</th>
      <td>(g) -25 to 50</td>
      <td>(h) -25 to 25</td>
      <td>(i) -25 to 25</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(g) -75 to 75</td>
      <td>(h) -50 to 50</td>
      <td>(i) -50 to 50</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(g) -50 to 75</td>
      <td>(h) -100 to 100</td>
      <td>(i) -50 to 50</td>
    </tr>
    <tr>
      <th rowspan="3">30天 ($\hat{\sigma}_t^{30}$)</th>
      <th>價內 (ITM)</th>
      <td>(j) -50 to 75</td>
      <td>(k) -50 to 50</td>
      <td>(l) -25 to 25</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(j) -100 to 100</td>
      <td>(k) -75 to 75</td>
      <td>(l) -75 to 75</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(j) -75 to 100</td>
      <td>(k) -100 to 100</td>
      <td>(l) -75 to 75</td>
    </tr>
    <tr>
      <th rowspan="3">10天 ($\hat{\sigma}_t^{10}$)</th>
      <th>價內 (ITM)</th>
      <td>(m) -50 to 50</td>
      <td>(n) -50 to 50</td>
      <td>(o) -25 to 50</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(m) -100 to 75</td>
      <td>(n) -75 to 75</td>
      <td>(o) -75 to 75</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(m) -75 to 100</td>
      <td>(n) -100 to 100</td>
      <td>(o) -75 to 100</td>
    </tr>
    <tr>
      <th rowspan="3">21天 ($\hat{\sigma}_t^{21}$)</th>
      <th>價內 (ITM)</th>
      <td>(p) -50 to 50</td>
      <td>(q) -50 to 50</td>
      <td>(r) -25 to 50</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(p) -75 to 75</td>
      <td>(q) -75 to 75</td>
      <td>(r) -75 to 75</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(p) -75 to 100</td>
      <td>(q) -100 to 100</td>
      <td>(r) -75 to 100</td>
    </tr>
    <tr>
      <th rowspan="3">252天 ($\hat{\sigma}_t^{252}$)</th>
      <th>價內 (ITM)</th>
      <td>(s) -25 to 50</td>
      <td>(t) -25 to 75</td>
      <td>(u) -25 to 75</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(s) -50 to 100</td>
      <td>(t) -75 to 100</td>
      <td>(u) -50 to 100</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(s) -75 to 75</td>
      <td>(t) -50 to 50</td>
      <td>(u) -75 to 100</td>
    </tr>
    <tr>
      <th rowspan="3">波動率指數 (VBI) ($\hat{\sigma}_t^{vbi}$)</th>
      <th>價內 (ITM)</th>
      <td>(v) -50 to 50</td>
      <td>(w) -25 to 50</td>
      <td>(x) -25 to 50</td>
    </tr>
    <tr>
      <th>價平 (ATM)</th>
      <td>(v) -100 to 50</td>
      <td>(w) -50 to 100</td>
      <td>(x) -50 to 50</td>
    </tr>
    <tr>
      <th>價外 (OTM)</th>
      <td>(v) -75 to 50</td>
      <td>(w) -75 to 75</td>
      <td>(x) -75 to 50</td>
    </tr>
  </tbody>
</table>

### 圖 6. 依據履約價格比率分類之模型定價誤差（賣權）：全樣本。

<table>
  <caption>圖 7. BIST 30 指數選擇權定價誤差分布 — 面板 (a) 至 (x) 依模型、波動率預測方法及價內程度 (ITM、ATM、OTM) 分類</caption>
  <thead>
    <tr>
      <th rowspan="2">波動率預測方法</th>
      <th rowspan="2">價內程度</th>
      <th colspan="3">定價誤差範圍 (約略)</th>
    </tr>
    <tr>
      <th>BS 模型</th>
      <th>NN1 模型</th>
      <th>NN2 模型</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3">(a-c) $\hat{\sigma}_t^{implied}$</th>
      <th>ITM</th>
      <td>-2 to 0</td>
      <td>-2 to 1</td>
      <td>-1 to 1</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 1</td>
      <td>-1 to 2</td>
      <td>-2 to 2</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-5 to 0</td>
      <td>-3 to 3</td>
      <td>-2 to 5</td>
    </tr>
    <tr>
      <th rowspan="3">(d-f) $\hat{\sigma}_t^{garch}$</th>
      <th>ITM</th>
      <td>-2 to 0</td>
      <td>-1 to 1</td>
      <td>-1 to 1</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-2 to 2</td>
      <td>-1 to 2</td>
      <td>-1 to 2</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-3 to 1</td>
      <td>-1 to 3</td>
      <td>-1 to 6</td>
    </tr>
    <tr>
      <th rowspan="3">(g-i) $\hat{\sigma}_t^{360}$</th>
      <th>ITM</th>
      <td>-3 to 0</td>
      <td>-1 to 1</td>
      <td>-1 to 1</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 1</td>
      <td>-2 to 2</td>
      <td>-2 to 2</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-4 to 1</td>
      <td>-3 to 3</td>
      <td>-1 to 6</td>
    </tr>
    <tr>
      <th rowspan="3">(j-l) $\hat{\sigma}_t^{30}$</th>
      <th>ITM</th>
      <td>-2 to 2</td>
      <td>-1 to 2</td>
      <td>-1 to 2</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 2</td>
      <td>-1 to 3</td>
      <td>-2 to 3</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-4 to 2</td>
      <td>-1 to 3</td>
      <td>-2 to 6</td>
    </tr>
    <tr>
      <th rowspan="3">(m-o) $\hat{\sigma}_t^{10}$</th>
      <th>ITM</th>
      <td>-3 to 1</td>
      <td>-1 to 2</td>
      <td>-1 to 1</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 2</td>
      <td>-2 to 2</td>
      <td>-2 to 2</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-5 to 2</td>
      <td>-3 to 3</td>
      <td>-2 to 5</td>
    </tr>
    <tr>
      <th rowspan="3">(p-r) $\hat{\sigma}_t^{21}$</th>
      <th>ITM</th>
      <td>-3 to 1</td>
      <td>-1 to 2</td>
      <td>-1 to 2</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 2</td>
      <td>-1 to 3</td>
      <td>-2 to 3</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-4 to 2</td>
      <td>-1 to 4</td>
      <td>-1 to 6</td>
    </tr>
    <tr>
      <th rowspan="3">(s-u) $\hat{\sigma}_t^{252}$</th>
      <th>ITM</th>
      <td>-3 to 1</td>
      <td>0 to 3</td>
      <td>-1 to 2</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 2</td>
      <td>-1 to 3</td>
      <td>-1 to 4</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-4 to 2</td>
      <td>0 to 4</td>
      <td>-1 to 7</td>
    </tr>
    <tr>
      <th rowspan="3">(v-x) $\hat{\sigma}_t^{vbi}$</th>
      <th>ITM</th>
      <td>-3 to 1</td>
      <td>-1 to 1</td>
      <td>-1 to 2</td>
    </tr>
    <tr>
      <th>ATM</th>
      <td>-3 to 2</td>
      <td>-1 to 2</td>
      <td>-1 to 3</td>
    </tr>
    <tr>
      <th>OTM</th>
      <td>-4 to 2</td>
      <td>-2 to 3</td>
      <td>-1 to 5</td>
    </tr>
  </tbody>
</table>

圖 7. 依據履約價格比率分類之模型定價誤差（賣權）：子樣本。

<table>
  <caption>各模型與波動度預測方法之定價誤差——(a)至(x)面板；y軸：定價誤差；x軸：剩餘到期時間</caption>
  <thead>
    <tr>
      <th rowspan="2">波動度預測方法</th>
      <th colspan="3">模型</th>
    </tr>
    <tr>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$\hat{\sigma}_t^{implied}$</th>
      <td>(a) BS($\hat{\sigma}_t^{implied}$)</td>
      <td>(b) NN1($\hat{\sigma}_t^{implied}$)</td>
      <td>(c) NN2($\hat{\sigma}_t^{implied}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{garch}$</th>
      <td>(d) BS($\hat{\sigma}_t^{garch}$)</td>
      <td>(e) NN1($\hat{\sigma}_t^{garch}$)</td>
      <td>(f) NN2($\hat{\sigma}_t^{garch}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{360}$</th>
      <td>(g) BS($\hat{\sigma}_t^{360}$)</td>
      <td>(h) NN1($\hat{\sigma}_t^{360}$)</td>
      <td>(i) NN2($\hat{\sigma}_t^{360}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{30}$</th>
      <td>(j) BS($\hat{\sigma}_t^{30}$)</td>
      <td>(k) NN1($\hat{\sigma}_t^{30}$)</td>
      <td>(l) NN2($\hat{\sigma}_t^{30}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{10}$</th>
      <td>(m) BS($\hat{\sigma}_t^{10}$)</td>
      <td>(n) NN1($\hat{\sigma}_t^{10}$)</td>
      <td>(o) NN2($\hat{\sigma}_t^{10}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{21}$</th>
      <td>(p) BS($\hat{\sigma}_t^{21}$)</td>
      <td>(q) NN1($\hat{\sigma}_t^{21}$)</td>
      <td>(r) NN2($\hat{\sigma}_t^{21}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{252}$</th>
      <td>(s) BS($\hat{\sigma}_t^{252}$)</td>
      <td>(t) NN1($\hat{\sigma}_t^{252}$)</td>
      <td>(u) NN2($\hat{\sigma}_t^{252}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{vbi}$</th>
      <td>(v) BS($\hat{\sigma}_t^{vbi}$)</td>
      <td>(w) NN1($\hat{\sigma}_t^{vbi}$)</td>
      <td>(x) NN2($\hat{\sigma}_t^{vbi}$)</td>
    </tr>
  </tbody>
</table>

圖 8. 各模型依剩餘到期時間之定價誤差（賣權）：全樣本。

<table>
  <caption>依據不同模型與波動率預測方法，按價內程度與剩餘到期時間分類之定價誤差。面板(a)至(x)顯示以價內(ITM, y=140)、價平(ATM, y=90)及價外(OTM, y=30)選擇權的定價誤差（縱軸）對剩餘到期時間（橫軸）的圖形。</caption>
  <thead>
    <tr>
      <th rowspan="2">波動率預測方法</th>
      <th colspan="3">模型</th>
    </tr>
    <tr>
      <th>BS</th>
      <th>NN1</th>
      <th>NN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$\hat{\sigma}_t^{implied}$</th>
      <td>(a) BS($\hat{\sigma}_t^{implied}$)</td>
      <td>(b) NN1($\hat{\sigma}_t^{implied}$)</td>
      <td>(c) NN2($\hat{\sigma}_t^{implied}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{garch}$</th>
      <td>(d) BS($\hat{\sigma}_t^{garch}$)</td>
      <td>(e) NN1($\hat{\sigma}_t^{garch}$)</td>
      <td>(f) NN2($\hat{\sigma}_t^{garch}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{360}$</th>
      <td>(g) BS($\hat{\sigma}_t^{360}$)</td>
      <td>(h) NN1($\hat{\sigma}_t^{360}$)</td>
      <td>(i) NN2($\hat{\sigma}_t^{360}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{30}$</th>
      <td>(j) BS($\hat{\sigma}_t^{30}$)</td>
      <td>(k) NN1($\hat{\sigma}_t^{30}$)</td>
      <td>(l) NN2($\hat{\sigma}_t^{30}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{10}$</th>
      <td>(m) BS($\hat{\sigma}_t^{10}$)</td>
      <td>(n) NN1($\hat{\sigma}_t^{10}$)</td>
      <td>(o) NN2($\hat{\sigma}_t^{10}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{21}$</th>
      <td>(p) BS($\hat{\sigma}_t^{21}$)</td>
      <td>(q) NN1($\hat{\sigma}_t^{21}$)</td>
      <td>(r) NN2($\hat{\sigma}_t^{21}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{252}$</th>
      <td>(s) BS($\hat{\sigma}_t^{252}$)</td>
      <td>(t) NN1($\hat{\sigma}_t^{252}$)</td>
      <td>(u) NN2($\hat{\sigma}_t^{252}$)</td>
    </tr>
    <tr>
      <th>$\hat{\sigma}_t^{vbi}$</th>
      <td>(v) BS($\hat{\sigma}_t^{vbi}$)</td>
      <td>(w) NN1($\hat{\sigma}_t^{vbi}$)</td>
      <td>(x) NN2($\hat{\sigma}_t^{vbi}$)</td>
    </tr>
  </tbody>
</table>
<table>
  <caption>面板(a)至(x)定價誤差分布摘要</caption>
  <thead>
    <tr>
      <th>面板</th>
      <th>模型與波動率方法</th>
      <th>價內程度類別（縱軸）</th>
      <th>剩餘到期時間範圍（橫軸）</th>
      <th>定價誤差集中情形</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3">(a)</th>
      <td rowspan="3">BS($\hat{\sigma}_t^{implied}$)</td>
      <th>價內 (140)</th>
      <td>-5 to 5</td>
      <td>分散於0附近</td>
    </tr>
    <tr>
      <th>價平 (90)</th>
      <td>-5 to 5</td>
      <td>集中於0附近</td>
    </tr>
    <tr>
      <th>價外 (30)</th>
      <td>-5 to 5</td>
      <td>集中於0附近</td>
    </tr>
    <tr>
      <th rowspan="3">(b)</th>
      <td rowspan="3">NN1($\hat{\sigma}_t^{implied}$)</td>
      <th>價內 (140)</th>
      <td>-5 to 5</td>
      <td>緊密聚集於0</td>
    </tr>
    <tr>
      <th>價平 (90)</th>
      <td>-5 to 5</td>
      <td>緊密聚集於0</td>
    </tr>
    <tr>
      <th>價外 (30)</th>
      <td>-5 to 5</td>
      <td>緊密聚集於0</td>
    </tr>
    <tr>
      <th rowspan="3">(c)</th>
      <td rowspan="3">NN2($\hat{\sigma}_t^{implied}$)</td>
      <th>價內 (140)</th>
      <td>-5 to 5</td>
      <td>緊密聚集於0</td>
    </tr>
    <tr>
      <th>價平 (90)</th>
      <td>-5 to 5</td>
      <td>緊密聚集於0</td>
    </tr>
    <tr>
      <th>價外 (30)</th>
      <td>-5 to 5</td>
      <td>緊密聚集於0</td>
    </tr>
    <tr>
      <th colspan="5">... [面板(d)至(x)呈現類似視覺型態，僅在零誤差垂直線（x=0）周圍的離散程度有所差異] ...</th>
    </tr>
  </tbody>
</table>

圖 9. 依到期時間分類的模型定價誤差（賣權）：子樣本。

Z. İltüzer

Borsa İstanbul Review 22-4 (2022) 725–742

樣本與子樣本分析的結果。BS 模型在使用所有波動率預測方法時，對於短期、中期與長期的買權（call options）均傾向低估（underpricing），但使用隱含波動率（implied volatility）以及過去十天資料估計的歷史波動率（historical volatility）的情況除外。然而，NN2 模型在使用所有波動率方法時，均持續高估（overprices）短期買權；NN1 模型則除了使用過去十天資料估計的歷史波動率之外，並未顯示出明顯的高估或低估傾向。在<u>圖 5</u>所示的子樣本分析結果中，BS 的低估以及 NN1 與 NN2 的高估行為，在動盪時期對於短期與中期買權同樣存在。由於測試樣本中僅有一筆剩餘到期時間約 250 天的長期買權，因此將其從圖表中排除，以更清楚地呈現短期與中期買權的結果。

## 3.2. 賣權（Put options）

<mark>表 3 與表 4</mark>分別呈現各模型在全樣本與子樣本分析下，針對賣權在價內程度（moneyness）與剩餘到期時間（time-to-maturity）維度的樣本外均方根誤差（out-of-sample RMSEs）。在所有價內程度維度（即價外（OTM）、價平（ATM）與價內（ITM）賣權）中，最佳模型均為使用 360 天歷史波動率的 BS 模型。在剩餘到期時間維度方面，對於所有到期期間群組（即短期與中期賣權），最佳模型同樣是使用過去 360 天資料估計歷史波動率的 BS 模型。對於長期賣權，最佳模型為使用 VBI 的 BS，其次為使用過去 360 天資料估計歷史波動率的 BS（RMSE 值第二低）。在短期賣權方面，BS 在所有波動率預測方法下均為最佳模型。整體而言，在土耳其選擇權市場中，無論價內程度或剩餘到期時間為何，占主導地位的模型皆為使用過去 360 天資料估計歷史波動率的 BS 模型。在動盪時期，表現最佳的模型依序為：價外賣權使用 VBI 的 BS、價平賣權使用隱含波動率的 NN2（或 BS），以及價內賣權使用過去 360 天資料估計歷史波動率的 NN2。在正常與動盪時期，BS 皆是在所有波動率預測方法下對價外賣權的主導模型；而 NN 則是在動盪時期對價平與價內賣權的所有波動率預測方法下均為主導模型。在剩餘到期時間維度方面，使用隱含波動率的 NN1 是短期賣權的最佳模型，使用 GARCH 波動率的 NN2 則是中期賣權的最佳模型。由於動盪時期缺乏到期時間超過 90 天且 2018 年 4 月收盤價不為零的賣權，因此未報告長期賣權在動盪時期的結果。

**Output:**

本研究提供的結果與方法，可供實務工作者作為建置類神經網路（NN）模型以及選擇定價模型的指引，而無需進行詳細、計算密集且複雜的績效評估流程。然而，從子樣本分析中極小的 RMSE 值可知，子樣本分析進行的是單月期預測，而全樣本分析則進行八個月期預測，以涵蓋足夠的樣本外資料來代表不同價內程度（moneyness）與剩餘到期時間（time-to-maturity）維度的群組。因此最佳做法是每日更新樣本，將前一日資料加入後重新定價。

<u>圖 6 與圖 7</u> 分別呈現全樣本與子樣本分析中，各模型依價內程度分類的定價誤差（$p - \hat{p}$）。根據全樣本分析，BS 模型傾向高估賣權（put options），而 NN 模型則傾向低估賣權，此結果與買權（call options）的發現完全相反。子樣本分析亦呈現相同結果。具體而言，採用所有波動率預測的 NN2 僅傾向低估價外（OTM）選擇權，而 NN1 則低估價外（OTM）、價平（ATM）與價內（ITM）選擇權，僅在使用隱含波動率（implied volatility）預測以及過去 360 天歷史波動率估計值時例外。

<u>圖 8 與圖 9</u> 呈現全樣本與子樣本分析中，各模型依剩餘到期時間分類的定價誤差。根據全樣本結果，BS 模型在使用幾乎所有波動率預測方法時，皆傾向高估短期、中期與長期賣權，而 NN 模型則傾向低估中期與長期賣權。具體而言，採用所有波動率輸入的 NN2 會高估短期選擇權，同時低估中期與長期選擇權。子樣本分析顯示，BS 模型同樣高估短期與中期選擇權。NN1 模型在全樣本中低估短期與中期賣權，然而 NN2 模型除使用 360 天與 30 天歷史波動率之分析外，並未顯示明顯的低估或高估傾向。綜合考量<mark>表 3</mark>與<u>圖 6、圖 8</u>中各模型的 RMSE，整體而言，以過去 360 天資料估計歷史波動率的 BS 模型準確度最高，儘管其持續呈現高估傾向。

全樣本分析強烈顯示，傳統 BS 選擇權定價模型在賣權上的表現優於 NN 模型，而 NN2 模型在買權上的表現則優於 BS 模型與 NN1。此買權結果與文獻中針對較成熟股票市場（如 S&P、FTSE、DAX、日經指數與 OMX）使用 NN 進行選擇權定價的研究發現一致。然而在動盪時期，NN 模型對賣權的表現優於 BS 模型，而 BS 模型對買權的表現則優於 NN 模型。

## 4. 摘要與結論

金融衍生性商品的使用及其重要性日益提升，使經濟主體得以管理更為整合且波動劇烈的金融市場風險。這也促使選擇權定價文獻自 <u>Black and Scholes (1972)</u> 的開創性研究以來快速發展。此後，由於模型假設逐漸放寬，BS 模型已發展出許多不同版本。然而，在

740

Z. İltüzer

Borsa İstanbul Review 22-4 (2022) 725–742

過去二十年來，人工神經網路（neural networks, NNs）在選擇權定價上的表現已受到研究人員廣泛檢視與分析，特別是針對已開發股市指數選擇權，結果多顯示NNs優於傳統的Black-Scholes（BS）模型。然而，針對新興股市指數選擇權的研究仍付之闕如，這使得我們難以判斷NNs在已開發市場的定價優勢是否同樣適用於新興市場。本研究旨在填補此一缺口，針對BIST 30指數買權與賣權，在平穩時期與動盪時期，比較NNs與BS模型的定價表現。本研究亦探討不同的波動率預測方法——即GARCH、隱含波動率（implied volatility）、歷史波動率，以及隱含波動率指數（VBI）——是否會影響並提升模型的表現。

整體而言，在平穩時期，NN模型在買權定價上表現最佳，而BS模型在賣權定價上表現最佳；然而在動盪時期，無論在何種價內程度（moneyness）與剩餘到期時間（time-to-maturity）維度下，買權的最佳模型為BS，賣權的最佳模型則為NN。兩種模型在價內（ITM）買權的定價表現均不佳，其定價誤差在平穩與動盪時期皆為其他買權與賣權的四至五倍，與所使用模型無關。Black-Scholes模型在使用幾乎所有波動率預測方法時，在平穩與動盪時期皆呈現低估買權、高估賣權的偏誤。研究結果顯示，市場參與者對買權與賣權的看待角度幾乎完全相反，這可能是因為交易「買入權利」被市場參與者視為比交易「賣出權利」更具風險。未來值得深入研究的方向之一，是探討模型在價內買權上相對表現較差的原因，並針對此類選擇權發展適當的定價方法，這將可為土耳其選擇權市場的參與者帶來 valuable 的結果。

## **利益衝突聲明**

無。

## **參考文獻**

Ahir, H., Bloom, N., & Furceri, D. (2018). The world uncertainty index. https://ssrn.com/abstract=3275033.

Amilon, H. (2003). A neural network versus Black–Scholes: A comparison of pricing and hedging performances. Journal of Forecasting, 22, 317–335. https://doi.org/10.1002/for.867

Amin, K. I., & Ng, V. K. (1993). Option valuation with systematic stochastic volatility. The Journal of Finance, 48, 881–910. https://doi.org/10.1111/j.1540-6261.1993.tb04023.x

Anders, U., Korn, O., & Schmitt, C. (1998). Improving the pricing of options: A neural network approach. Journal of Forecasting, 17, 369–388. https://doi.org/10.1002/(SICI)1099-131X(1998090)17:5<369::AID-FOR702>3.0.CO;2-S

Bennell, J., & Sutcliffe, C. (2005). Black–Scholes versus artificial neural networks in pricing FTSE 100 options. Intelligent Systems, 12, 243–260. https://doi.org/10.1002/isaf.254

<font color="#0000FF">Black, F., & Scholes, M. (1972). 選擇權契約的評價與市場效率的測試（The valuation of option contracts and a test of market efficiency）。<font color="#0000FF">《財務學期刊》，27，399–417。<font color="#0000FF">http://www.jstor.org/stable/2978484。</font></font></font>

<font color="#0000FF">Bollerslev, T. (1986). 廣義自迴歸條件異方差（Generalized autoregressive conditional heteroskedasticity）。<font color="#0000FF">《計量經濟學期刊》，31，307–327。<font color="#0000FF">https://doi.org/10.1016/0304-4076(86)90063-1。<font color="#0000FF">https://www.sciencedirect.com/science/article/pii/0304407686900631</font></font></font></font>

<font color="#0000FF">Boyle, P. P. (1988). 具有兩個狀態變數的選擇權定價格狀架構（A lattice framework for option pricing with two state variables）。<font color="#0000FF">《財務與計量分析期刊》，23，1–12。<font color="#0000FF">https://doi.org/10.2307/2331019</font></font></font>

<font color="#0000FF">Chance, D. M., Hanson, T. A., Li, W., & Muthuswamy, J. (2017). 波動率微笑的偏差（A bias in the volatility smile）。<font color="#0000FF">《衍生性商品研究評論》，47，47–90。<font color="#0000FF">https://doi.org/10.1007/s11147-016-9124-0</font></font></font>

<font color="#0000FF">Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). 選擇權定價：一種簡化方法（Option pricing: A simplified approach）。<font color="#0000FF">《財務經濟學期刊》，7，229–263。<font color="#0000FF">https://doi.org/10.1016/0304-405X(79)90015-1。<font color="#0000FF">https://www.sciencedirect.com/science/article/pii/0304405X79900151</font></font></font></font>

<font color="#0000FF">Daglish, T. (2003). 美國指數選擇權參數與非參數方法之定價與避險比較（A pricing and hedging comparison of parametric and nonparametric approaches for American index options）。<font color="#0000FF">《財務計量經濟學期刊》，1，327–364。<font color="#0000FF">https://doi.org/10.1093/jjfinec/nbg015</font></font></font>

<font color="#0000FF">Duan, J. C. (1995). GARCH選擇權定價模型（The GARCH option pricing model）。<font color="#0000FF">《數理財務學》，5，13–32。<font color="#0000FF">https://doi.org/10.1111/j.1467-9965.1995.tb00099.x</font></font></font>

<font color="#0000FF">Dumas, B., Fleming, J., & Whaley, R. E. (2002). 隱含波動率函數：實證檢定（Implied volatility functions: Empirical tests）。<font color="#0000FF">《財務學期刊》，53，2059–2106。<font color="#0000FF">https://doi.org/10.1111/10022-1082.00083</font></font></font>

<font color="#0000FF">Fadda, S. (2020). 以雙重波動率輸入模組化神經網路進行選擇權定價（Pricing options with dual volatility input to modular neural networks）。<font color="#0000FF">《伊斯坦堡證交所評論》，20，269–278。<font color="#0000FF">https://doi.org/10.1016/j.bir.2020.03.002。<font color="#0000FF">https://www.sciencedirect.com/science/article/pii/S2214845020300168</font></font></font></font>

<font color="#0000FF">Garcia, R., & Gencay, R. (2000). 使用神經網路與齊次性提示對衍生性商品進行定價與避險（Pricing and hedging derivative securities with neural networks and a homogeneity hint）。<font color="#0000FF">《計量經濟學期刊》，94，93–115。<font color="#0000FF">https://doi.org/10.1016/S0304-4076(99)00018-4。<font color="#0000FF">https://www.sciencedirect.com/science/article/pii/S0304407699000184</font></font></font></font>

<font color="#0000FF">Gaspar, R. M., Lopes, S. D., & Sequeira, B. (2020). 神經網路對美國賣權的定價（Neural network pricing of American put options）。<font color="#0000FF">《風險》，8，<font color="#0000FF">https://doi.org/10.3390/risks8030073。<font color="#0000FF">https://www.mdpi.com/2227-9091/8/3/73</font></font></font></font>

<font color="#0000FF">Glorot, X., Bordes, A., & Bengio, Y. (2011). 深度稀疏整流神經網路。In G. Gordon, D. Dunson, & M. Dudik (Eds.), <font color="#0000FF">Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 315–323).</font> Fort Lauderdale, FL: PMLR. <font color="#0000FF">http://proceedings.mlr.press/v15/glorot11a.html</font></font>

<font color="#0000FF">Gradojevic, N., Gencay, R., & Kukolj, D. (2009). 以模組化神經網路進行選擇權定價。 <font color="#0000FF">IEEE Transactions on Neural Networks, 20, 626–637. <font color="#0000FF">https://doi.org/10.1109/TNN.2008.2011130</font></font></font>

<font color="#0000FF">Gu, S., Kelly, B., & Xiu, D. (2020). 透過機器學習的實證資產定價。 <font color="#0000FF">Review of Financial Studies, 33, 2223–2273. <font color="#0000FF">https://doi.org/10.1093/rfs/hhaa009. arXiv <font color="#0000FF">https://academic.oup.com/rfs/article-pdf/33/5/2223/33209812/hhaa009.pdf</font></font></font></font>

<font color="#0000FF">Hull, J., & White, A. (1987). 具有隨機波動率資產之選擇權定價。 <font color="#0000FF">The Journal of Finance, 42, 281–300. <font color="#0000FF">https://doi.org/10.1111/j.1540-6261.1987.tb02568.x</font></font></font>

<font color="#0000FF">Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). 以學習網路對衍生性證券進行定價與避險的非參數方法。 <font color="#0000FF">The Journal of Finance, 49, 851–889. <font color="#0000FF">https://doi.org/10.1111/j.1540-6261.1994.tb00081.x</font></font></font>

<font color="#0000FF">İltüzer Samur, Z., & Temur, G. T. (2009). 人工神經網路在選擇權定價的應用：S&P 100指數選擇權之案例。 <font color="#0000FF">World Academy of Science, Engineering and Technology, 54, 326–331.</font></font>

<font color="#0000FF">Ivaşcu, C. F. (2021). 使用機器學習進行選擇權定價。 <font color="#0000FF">Expert Systems with Applications, 163, 113799. <font color="#0000FF">https://doi.org/10.1016/j.eswa.2020.113799. <font color="#0000FF">https://www.sciencedirect.com/science/article/pii/S0957417420306187</font></font></font></font>

<font color="#0000FF">Lajbcygier, P. (2004). 以乘積限制混合神經網路改善選擇權定價。 <font color="#0000FF">IEEE Transactions on Neural Networks, 15, 465–476. <font color="#0000FF">https://doi.org/10.1109/TNN.2004.824265</font></font></font>

<font color="#0000FF">Lin, C. T., & Yeh, H. Y. (2005). 台灣股票指數選擇權價格之評價：Black-Scholes模型與神經網路模型之績效比較。 <font color="#0000FF">Journal of Statistics & Management Systems, 8, 355–367. <font color="#0000FF">https://doi.org/10.1080/09720510.2005.10701164</font></font></font>

741

Z. İltüzer

*Borsa İstanbul Review 22-4 (2022) 725–742*

Lin, C. T., & Yeh, H. Y. (2009). Empirical of the Taiwan stock index option price forecasting model–applied artificial neural network. <u>Applied Economics</u>, 41, 1965–1972. <https://doi.org/10.1080/00036840601131672>

Macbeth, J. D., & Merville, L. J. (1979). An empirical examination of the Black-Scholes call option pricing model. <u>The Journal of Finance</u>, 34, 1173–1186. <http://www.jstor.org/stable/2327242>.

Malliaris, M., & Salchenberger, L. (1993). A neural network model for estimating option prices. <u>Journal of Applied Intelligence</u>, 3, 193–206. <https://doi.org/10.1007/BF00871937>

Morelli, M. J., Montagna, G., Nicrosini, O., Treccani, M., Farina, M., & Amato, P. (2004). Pricing financial derivatives with neural networks. <u>Physica A: Statistical Mechanics and its Applications</u>, 338, 160–165. <https://doi.org/10.1016/j.physa.2004.02.038>. <https://www.sciencedirect.com/science/article/pii/S037843710400233X>

Naik, V. (1993). Option valuation and hedging strategies with jumps in the volatility of asset returns. <u>The Journal of Finance</u>, 48, 1969–1984. <https://doi.org/10.1111/j.1540-6261.1993.tb05137.x>

<Poon, S. H. (2005). *A practical guide to forecasting financial market volatility.* England: Wiley Finance.

Rendleman, R. J., & Bartter, B. J. (1979). Two-state option pricing. <u>The Journal of Finance</u>, 34, 1093–1110. <http://www.jstor.org/stable/2327237>.

Rubinstein, M. (1983). Displaced diffusion option pricing. <u>The Journal of Finance</u>, 38, 213–217. <https://doi.org/10.1111/j.1540-6261.1983.tb03636.x>

Scott, L. O. (1987). Option pricing when the variance changes randomly: Theory, estimation, and an application. <u>Journal of Financial and Quantitative Analysis</u>, 22, 419–438. <https://doi.org/10.2307/2330793>

Scott, L. O. (2002). Pricing stock options in a jump-diffusion model with stochastic volatility and interest rates: Applications of Fourier inversion methods. *Mathematical Finance*, 7, 413–426. <https://doi.org/10.1111/1467-9965.00039>

Sensoy, A. S., & Omole, J. O. (2018). Implied volatility indices: A review and extension in the Turkish case. *International Review of Financial Analysis*, 60, 151–161. <https://doi.org/10.1016/j.irfa.2018.08.006>. <https://www.sciencedirect.com/science/article/pii/S1057521918305969>

Tseng, C. H., Cheng, S. T., Wang, Y. H., & Peng, J. T. (2008). Artificial neural network model of the hybrid EGARCH volatility of the Taiwan stock index option prices. *Physica A: Statistical Mechanics and its Applications*, 387, 343192–343200. <https://doi.org/10.1016/j.physa.2008.01.074>. <https://www.sciencedirect.com/science/article/pii/S0378437108000320>

Wang, Y. H. (2009a). Nonlinear neural network forecasting model for stock index option price: Hybrid GJR–GARCH approach. *Expert Systems with Applications*, 36, 564–570. <https://doi.org/10.1016/j.eswa.2007.09.056>. <https://www.sciencedirect.com/science/article/pii/S0957417407004654>

Wang, Y. H. (2009b). Using neural network to forecast stock index option price: A new hybrid GARCH approach. *Quality and Quantity*, 43, 833–843. <https://doi.org/10.1007/s11135-008-9176-9>

Wang, C. P., Lin, S. H., Huang, H. H., & Wu, P. C. (2012). Using neural network for forecasting TXO price under different volatility models. *Expert Systems with Applications*, 39, 5025–5032. <https://doi.org/10.1016/j.eswa.2011.11.038>. <https://www.sciencedirect.com/science/article/pii/S0957417411015818>

Yadav, K. (2021). Formulation of a rational option pricing model using artificial neural networks. *SoutheastCon*, 2021, 1–8. <https://doi.org/10.1109/SoutheastCon45413.2021.9401835>

Yang, Y., Zheng, Y., & Hospedales, T. (2017). Gated neural networks for option pricing: Rationality by design. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 31). <https://ojs.aaai.org/index.php/AAAI/article/view/10505>.

Yao, J., Li, Y., & Tan, C. L. (2000). Option price forecasting using neural networks. *Omega*, 28, 455–466. <https://doi.org/10.1016/S0305-0483(99)00066-3>. <https://www.sciencedirect.com/science/article/pii/S0305048399000663>

Zaheer, R., & Shaziya, H. (2018). GPU-based empirical evaluation of activation functions in convolutional neural networks. In *2018 2nd international conference on inventive systems and control* (pp. 769–773). ICISC. <https://doi.org/10.1109/ICISC.2018.8398903>.

742
