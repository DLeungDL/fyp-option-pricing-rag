# Ruf_Wang_Literature_Review

> 原文檔案：`Ruf_Wang_Literature_Review.md`  
> 語言：繁體中文（臺灣，zh-TW）  
> 說明：由 LlamaParse Markdown 分段機器翻譯；公式、表格數字與檔名未改寫。

---

# **神經網路（neural networks）於選擇權評價（option pricing）與避險（hedging）之應用：文獻回顧（literature review）**

**本論文於 LSE Research Online 的網址：** http://eprints.lse.ac.uk/104341/

版本：接受版（Accepted Version）

---

### **文章：**

Ruf, Johannes ORCID: 0000-0003-3616-2194 與 Wang, Weiguan (2020) Neural networks for option pricing and hedging: a literature review. Journal of Computational Finance, 24 (1). 1 - 46. ISSN 1460-1559

10.21314/JCF.2020.390

---

### **再利用**

存放於 LSE Research Online 之文件均受著作權保護，除另有標示外，保留所有權利。使用者得為私人研究目的下載及（或）列印該等文件，或從事各國著作權法所允許之其他行為。出版者或其他權利持有人或許允許對全文版本進行進一步之重製與再利用，此等授權資訊標示於該文件之 LSE Research Online 紀錄中。

# 類神經網路於選擇權定價與避險之應用：文獻回顧

Johannes Ruf* Weiguan Wang†

2020年5月8日

## **摘要**

自1990年代初期以來，類神經網路即被用作選擇權定價與避險的無母數方法（nonparametric method）。關於此主題所發表的論文已遠超過一百篇。本筆記旨在提供一個全面的回顧。我們就輸入特徵（input features）、輸出變數（output variables）、基準模型（benchmark models）、績效衡量指標（performance measures）、資料分割方法（data partition methods）與標的資產（underlying assets）等面向對各篇論文進行比較。此外，本文亦討論相關研究與正規化技術（regularisation techniques）。

## 1 緒論

自Malliaris and Salchenberger [1993b]與Hutchinson et al. [1994]以降，學術文獻中已有一百餘篇論文探討人工類神經網路（artificial neural networks, ANNs）在選擇權定價與避險上的應用。本文即為此一文獻之回顧。撰寫本摘要的動機源自我們的姊妹作Ruf and Wang [2020]。在該文中，我們延續本筆記的討論，特別是關於以歷史金融資料訓練類神經網路時，可能產生問題的資料洩漏（data leakage）。

線性迴歸模型（linear regression model）可視為一種將某輸入 $x$ 映射至輸出 $y$ 的仿射函數（affine function）。同樣地，類神經網路可視為線性與非線性函數的（可能重複的）複合函數，同樣將某輸入 $x$ 映射至輸出 $y$。訓練類神經網路通常對應於選擇其中的線性部分，使得此映射在某種意義下，對給定資料集（*訓練集*，training set）$(x_i, y_i)_i$（或其子集）而言是最佳的。最佳性通常以*損失函數*（loss function）來衡量，其衡量類神經網路輸出與給定資料之間的距離。

Stone-Weierstrass定理指出，緊緻集（compact set）上的任何連續函數皆可由多項式逼近。同樣地，*萬能逼近定理*（universal approximation theorems）確保類神經網路能以適當的方式逼近連續函數。特別是，類神經網路能夠捕捉輸入與輸出之間的非線性相依關係。

基於此理解，類神經網路可用於許多與選擇權定價及避險相關的應用。最常見的形式是，類神經網路學習選擇權價格，將其視為標的價格、履約價格（strike price）以及其他可能相關選擇權特性的函數。同樣地，類神經網路亦可被訓練以學習隱含波動度曲面（implied volatility surface）或最適避險比率（optimal hedging ratios）。在定價任務中，對應的損失函數通常選為觀察（模擬）選擇權價格與類神經網路預測價格之間的平方距離。在避險任務中，則比較觀察（模擬）選擇權價格與類神經網路避險投資組合的價值。

讓我們在定價任務的脈絡下提供一個正式的例子，即一個具有線性輸出的雙隱藏層（two-hidden layer）類神經網路。此種架構將輸入 $x$（通常為由數個特徵組成的向量，例如價內外程度（moneyness）、契約特定隱含波動度等）映射至輸出 $y$（選擇權價格），如下所示：

$$y = w_2 \cdot \phi(w_1 \cdot x).$$

---

我們感謝Agostino Capponi、Marc Chataigner、Stéphane Crépey、Antoine Jacquier與Martin Larsson對本筆記早期版本所提供的意見。

* 倫敦政治經濟學院數學系。電子郵件：j.ruf@lse.ac.uk
† 倫敦政治經濟學院數學系。電子郵件：w.wang34@lse.ac.uk

1

此處 $\phi$ 是一個非線性函數(即所謂的*激活函數*(activation function)),$w_1, w_2$ 是權重向量(weight vectors),而點號表示純量積(scalar product)。訓練這樣一個人工神經網路(artificial neural network, ANN),相當於尋找權重向量 $\hat{w}_1, \hat{w}_2$,使得 ANN 的輸出 $\hat{y}$ 對資料中某個子集(即*訓練集*(training set))內的所有樣本都接近選擇權價格 $y$。如前所述,衡量「接近」程度的常用準則是均方誤差(mean squared error)。

本文所討論的論文大多研究這種以 ANN 進行的近似在模擬資料集或真實資料集上的表現。這些論文採用不同的績效衡量指標(performance measures),且常將 ANN 與各種基準(benchmarks)比較,其中最簡單的基準是 Black-Scholes 公式。我們也將概述各篇論文如何選擇訓練資料。

通用逼近定理(universal approximation theorems)讓我們能以「以模型為基礎」的方式使用 ANN。想像有一個資料產生過程(data-generating process),搭配一個計算上相當繁複的定價演算法,例如依賴求解偏微分方程(partial differential equations)或蒙地卡羅模擬(Monte-Carlo simulations)。面對這種情況時,可以利用 ANN 直接學習定價公式。我們將在第 4 節回顧這方面的文獻。

本文的組織架構如下。第 2 節呈現表 1,彙整有關使用 ANN 對選擇權進行無母數(nonparametric)定價(與避險)的文獻。第 3 節提供從表 1 中挑選出的推薦論文清單。第 4 節概述將 ANN 應用於選擇權定價與避險、但未必作為無母數估計工具的相關研究。第 5 節簡要討論所回顧文獻中使用的各種正規化(regularisation)技術。

## 2 文獻中基於 ANN 的選擇權定價與避險

Bennell and Sutcliffe [2004]、Chen and Sutcliffe [2012] 以及 Hahn [2013]<sup>1</sup> 針對 ANN 應用於選擇權定價與避險問題提供了廣泛的文獻回顧。在此,我們以其他以及更近期的研究來補充這些回顧。

表 1 彙整了大部分的文獻,並比較六項相關特性,分別為:特徵(或所謂的解釋變數(explanatory variables))、ANN 的輸出、基準模型(benchmark models)、訓練集與測試集之間的資料劃分,以及標的資產(underlyings)與資料的時間跨度。在表 1 中,我們僅列出以略帶統計觀點研究 ANN 在選擇權定價與避險問題上表現的論文。其他論文採用不同的取向,例如計算的觀點,因此並不適合直接納入表中。這些論文將於第 4 節另行討論。

我們並未納入參數估計方法的比較,也未比較 ANN 的架構,例如節點與層數、激活函數等。這些設定在此處所彙整的論文之間差異很大。至於整體趨勢,我們僅指出:較新的論文使用更複雜的架構,這與計算資源可取得性的提升相符。我們也未逐篇摘錄各論文所得出的具體結論。不過,超過半數論文的摘要明確強調 ANN 在選擇權定價與避險任務上的良好表現。

讓我們說明如何閱讀表 1。該表彙整了六項相關特徵，用以描述每篇論文如何處理定價/避險問題。「特徵」（Features）與「輸出」（Outputs）兩欄分別顯示提供給人工神經網路（artificial neural network, ANN）作為輸入的解釋特徵以及其輸出。表 2 說明這些欄位所使用的記號與縮寫。「基準」（Benchmarks）欄列出用來與 ANN 比較的非 ANN 技術，表 3 說明對應的縮寫。表 4 提供「績效衡量指標」（Performance measures）欄的縮寫與定義，該欄彙整每篇論文中如何評估 ANN（及其基準）。以粗體標示的績效衡量指標與跨多期的評估相關。表 5 說明各研究中所使用、且列於「標的資產」（Underlyings）欄之標的資產的縮寫。

以下是表 1 的「執行摘要」：

- 將股價與選擇權履約價作為 ANN 的輸入有兩種方式。有時它們被當作兩個分開的特徵；其他情況下，則僅使用兩者的比值（即所謂的價內外程度，moneyness）作為

<sup>1</sup> Hahn [2013] 亦回顧了利用 ANN 預測已實現波動度（realized volatility）的文獻，本文的目的並不在此。

2

作為輸入。在過去十年間，第二種方法較常被使用。關於此點的討論，另見第 2.1 小節。

- 在輸入特徵（input features）與基準（benchmark）方面，波動率估計（volatility estimates）有許多不同的選擇。所得出的結論往往取決於這項選擇。第 2.1 與 2.3 小節對此點提供更多細節。

- 大多數論文聚焦於估計選擇權價格（option prices），約有十五篇論文（佔所列論文的 10%）聚焦於估計隱含波動率（implied volatilities），僅極少數直接處理避險（hedging）問題；另見第 2.2 小節。

- 在某些研究中，資料被劃分為訓練集（training set）與測試集（test set）的方式違反了底層的時間序列（time series）結構。這會造成資訊洩漏（information leakage），並低估人工神經網路（artificial neural network, ANN）的泛化誤差（generalization error）。第 2.4 小節對此有進一步的討論。

對於只想從上述所有論文中選讀少數幾篇的讀者，我們建議參閱第 3 節。

在閱讀約 150 篇論文並製作表 1 之後，我們想就實作人工神經網路作為選擇權價格與避險之無母數（nonparametric）估計工具一事，提出三項（個人）建議。第一，應以定態（stationary）特徵作為輸入。第二，人工神經網路的表現應與適當的基準進行比較。第三，在將資料集劃分為訓練集與測試集時，不應違反時間序列結構。

3

<table>
  <thead>
    <tr>
      <th>作者 & 年份</th>
      <th>特徵</th>
      <th>輸出</th>
      <th>基準模型</th>
      <th>績效衡量指標</th>
      <th>資料分割方法</th>
      <th>標的資產</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Malliaris and Salchenberger [1993a,b]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>IM</sub>, <i>r</i>, 落後期 <i>C</i> 與 <i>S</i></td>
      <td><i>C</i></td>
      <td>BS-IM</td>
      <td>MAE, MAPE, MSE</td>
      <td>時間順序 (Chronological)</td>
      <td>S&amp;P100. 6個月</td>
    </tr>
    <tr>
      <td>Hutchinson et al. [1994]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-H, Linear</td>
      <td><strong>MATE</strong>, <strong>PE</strong>, <i>R</i><sup>2</sup></td>
      <td>時間順序 (Chronological)</td>
      <td>模擬資料 (BS)；S&amp;P500. 5年</td>
    </tr>
    <tr>
      <td>Kelly [1994]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>CRR</td>
      <td>MAE, <strong>MTE</strong>, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>個別股票. 6個月</td>
    </tr>
    <tr>
      <td>Boek et al. [1995]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td>(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i></td>
      <td>BS-H</td>
      <td>MAPE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>AOSPI. 2年</td>
    </tr>
    <tr>
      <td>Miranda and Burgess [1995]</td>
      <td>?</td>
      <td>Δ<i>σ</i><sub>I</sub></td>
      <td>Linear</td>
      <td>?</td>
      <td>?</td>
      <td>IBEX35. ?</td>
    </tr>
    <tr>
      <td>Krause [1996]</td>
      <td><i>C</i><sub>BS−H</sub>, <i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td><i>R</i><sup>2</sup></td>
      <td>時間順序 (Chronological)</td>
      <td>DAX. 3年</td>
    </tr>
    <tr>
      <td>Lachtermacher and Rodrigues Gaspar [1996]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MAE, MAPE, MPE, MSE</td>
      <td>隨機分割 (Random)</td>
      <td>個別股票. 2個月</td>
    </tr>
    <tr>
      <td>Lajbcygier and Flitman [1996]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>IH</sub></td>
      <td>(<i>C</i> − <i>C</i><sub>BS−IH</sub>)/<i>K</i></td>
      <td>BS-IH, KR, Linear</td>
      <td>MAE, <i>R</i><sup>2</sup></td>
      <td>時間順序 (Chronological)</td>
      <td>AOSPI. 3年</td>
    </tr>
    <tr>
      <td>Lajbcygier et al. [1996a]<sup>2</sup></td>
      <td>?</td>
      <td>?</td>
      <td>BS-?, BW</td>
      <td>?</td>
      <td>?</td>
      <td>AOSPI. ?</td>
    </tr>
    <tr>
      <td>Lajbcygier et al. [1996b], Lajbcygier [2002]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-H, BW, Linear</td>
      <td>MAPE/MAE, MSE, <i>R</i><sup>2</sup></td>
      <td>隨機分割 (Random)</td>
      <td>AOSPI. 2年</td>
    </tr>
    <tr>
      <td>Liu [1996]</td>
      <td><i>S</i></td>
      <td><i>S</i><sup>3</sup></td>
      <td>BS-H</td>
      <td>MAE, MAX, MSE</td>
      <td>時間順序 (Chronological)</td>
      <td>S&amp;P500. 5年<sup>4</sup></td>
    </tr>
    <tr>
      <td>Malliaris and Salchenberger [1996]</td>
      <td><i>τ</i>, 落後期 <i>σ</i><sub>IM</sub>, 及其他</td>
      <td><i>σ</i><sub>IM</sub></td>
      <td>無</td>
      <td>MAE, MSE</td>
      <td>時間順序 (Chronologi</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>作者</th>
      <th>輸入變數</th>
      <th>輸出變數</th>
      <th>比較基準</th>
      <th>績效衡量</th>
      <th>樣本分割</th>
      <th>資料集</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Niranjan [1996]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-H</td>
      <td>MSE</td>
      <td>?</td>
      <td>FTSE100. 11M</td>
    </tr>
    <tr>
      <td>Qi and Maddala [1996]<sup>5</sup></td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>r</i>, open interest</td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MAE, MSE, <i>R</i><sup>2</sup></td>
      <td>Random</td>
      <td>S&amp;P500. 2M</td>
    </tr>
    <tr>
      <td>Hanke [1997]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>G</sub>,<sup>6</sup> <i>r</i></td>
      <td><i>C</i>/<i>K</i>, (<i>C</i> − <i>C</i><sub>BS−G</sub>)/<i>K</i></td>
      <td>None</td>
      <td>MSE</td>
      <td>Chronological</td>
      <td>模擬 (SV)</td>
    </tr>
    <tr>
      <td>Herrmann and Narr [1997]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>σ</i><sub>V</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>BS-V</td>
      <td>MAE, ME, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>模擬 (BS)；DAX. 1Y</td>
    </tr>
    <tr>
      <td>Karaali et al. [1997]</td>
      <td><i>S</i>, <i>K</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>None</td>
      <td>None</td>
      <td>Chronological</td>
      <td>DEM 波動率. 5Y</td>
    </tr>
  </tbody>
</table>

<sup>2</sup>我們無法取得該論文的複本。
<sup>3</sup>該網路會逐步學習標的資產的動態，然後依賴蒙地卡羅模擬（Monte-Carlo）來決定選擇權價格。
<sup>4</sup>該網路以五年長度的股價路徑進行訓練，但僅使用單一交易日的選擇權價格資料。
<sup>5</sup>本文依賴齊（Qi）的博士論文 [1996]。
<sup>6</sup>此外也將其他 GARCH 參數加入作為特徵。

5

<table>
  <thead>
    <tr>
      <th>作者 & 年份</th>
      <th>特徵</th>
      <th>輸出</th>
      <th>基準模型</th>
      <th>績效衡量指標</th>
      <th>資料分割方法</th>
      <th>標的資產</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Lajbcygier and Connor [1997a,b]</td>
      <td><i>S/K, τ, σ</i><sub>IH</sub></td>
      <td>(<i>C − C</i><sub>BS−IH</sub>)/<i>K</i></td>
      <td>BS-IH</td>
      <td>MAE, SR</td>
      <td>時間順序</td>
      <td>AOSPI. 1年</td>
    </tr>
    <tr>
      <td>Lajbcygier et al. [1997]<sup>2</sup></td>
      <td><i>S/K</i>, ?</td>
      <td>(<i>C − C</i><sub>BS−?</sub>)/<i>K</i></td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>AOSPI. ?</td>
    </tr>
    <tr>
      <td>Ahmed and Swidler [1998]</td>
      <td><i>S/K, τ, σ</i><sub>H</sub>, 成交量</td>
      <td><i>σ</i><sub>I</sub></td>
      <td>無</td>
      <td>MAE, MSE</td>
      <td>隨機</td>
      <td>個別股票。3年</td>
    </tr>
    <tr>
      <td>Anders et al. [1998]</td>
      <td><i>S/K, S, τ, σ</i><sub>H</sub>, <i>σ</i><sub>V</sub>, <i>r</i></td>
      <td><i>C/K</i>, (<i>C − C</i><sub>BS−V</sub>)/<i>K</i></td>
      <td>BS-H, BS-V</td>
      <td>MAE, MAPE, ME, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>DAX。3年</td>
    </tr>
    <tr>
      <td>Avellaneda et al. [1998]</td>
      <td><i>S/K, τ</i></td>
      <td><i>σ</i><sub>I</sub></td>
      <td>無</td>
      <td>%E</td>
      <td>?</td>
      <td>USD-DEM。數天</td>
    </tr>
    <tr>
      <td>Garcia and Gençay [1998, 2000]</td>
      <td><i>S/K, τ</i></td>
      <td><i>C/K</i></td>
      <td>BS-H, 線性模型</td>
      <td>DM, <b>MATE</b>, MSE</td>
      <td>時間順序</td>
      <td>模擬 (BS)；S&amp;P500。8年</td>
    </tr>
    <tr>
      <td>White [1998]</td>
      <td>?</td>
      <td><i>C</i></td>
      <td>無</td>
      <td>MAE, MSE</td>
      <td>隨機</td>
      <td>模擬 (BS)</td>
    </tr>
    <tr>
      <td>Chen and Lee [1999]</td>
      <td><i>S, τ, σ</i><sub>H</sub>, Γ, Δ, <i>ρ</i>, 𝒱, 成交量</td>
      <td><i>C</i></td>
      <td>BS-H, CRR</td>
      <td>MAE, MAPE, MSE</td>
      <td>時間順序</td>
      <td>個別股票。1年</td>
    </tr>
    <tr>
      <td>Geigle and Aronson [1999]<sup>7</sup></td>
      <td><i>S/K, τ, σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C/K</i></td>
      <td>BS-H</td>
      <td>MAE, MAPE</td>
      <td>時間順序</td>
      <td>S&amp;P500。6年</td>
    </tr>
    <tr>
      <td>Hanke [1999a]</td>
      <td><i>S/K</i></td>
      <td>(<i>C − C</i><sub>BS−H</sub>)/<i>K</i></td>
      <td>BS-H</td>
      <td>MSE</td>
      <td>時間順序</td>
      <td>DAX。1年</td>
    </tr>
    <tr>
      <td>Hanke [1999b]</td>
      <td><i>S/K, τ, σ</i><sub>Cal</sub></td>
      <td><i>C/K</i>, (<i>C − C</i><sub>BS−Cal</sub>)/<i>K</i></td>
      <td>BS-Cal</td>
      <td>MSE</td>
      <td>時間順序</td>
      <td>DAX。10個月</td>
    </tr>
    <tr>
      <td>Ormoneit [1999]</td>
      <td><i>S/K</i></td>
      <td><i>C/K</i></td>
      <td>BS-H, BS-IH</td>
      <td><b>MATE</b>, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>DAX。9個月</td>
    </tr>
    <tr>
      <td>Tsaih [1999]</td>
      <td><i>S, K, τ, σ</i><sub>I</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>BS-IH</td>
      <td>敏感度分析</td>
      <td>時間順序</td>
      <td>模擬 (BS)</td>
    </tr>
    <tr>
      <td>Briegel and Tresp [2000]</td>
</tbody>
</table>

<table>
  <thead>
    <tr>
      <th>作者</th>
      <th>輸入變數</th>
      <th>輸出變數</th>
      <th>基準模型</th>
      <th>評估指標</th>
      <th>資料分割</th>
      <th>資料集</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Hutchinson et al. [1994]</td>
      <td><i>S, τ</i></td>
      <td><i>C</i></td>
      <td>BS-?</td>
      <td>MSE</td>
      <td>?</td>
      <td>FTSE100. 10M</td>
    </tr>
    <tr>
      <td>Carelli et al. [2000]</td>
      <td><i>K, τ</i></td>
      <td><i>σ</i><sub>I</sub></td>
      <td>None</td>
      <td>%E</td>
      <td>?</td>
      <td>USD-DEM. Several days</td>
    </tr>
    <tr>
      <td>de Freitas et al. [2000a,b]</td>
      <td><i>S/K, τ</i></td>
      <td><i>C/K</i></td>
      <td>BS-H</td>
      <td><i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>FTSE100. 11M</td>
    </tr>
    <tr>
      <td>Galindo-Flores [2000]</td>
      <td><i>S, K, τ</i></td>
      <td><i>C</i></td>
      <td>Decision tree, Linear, Nearest neighbour</td>
      <td>MSE</td>
      <td>?</td>
      <td>Simulation (BS)</td>
    </tr>
    <tr>
      <td>Ghaziri et al. [2000]</td>
      <td><i>S, K, τ, σ</i><sub>H</sub>, <i>r</i>, open interest</td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MSE</td>
      <td>?</td>
      <td>S&amp;P500. 2M</td>
    </tr>
    <tr>
      <td>Raberto et al. [2000]</td>
      <td><i>S/K, τ</i>, |<i>S − K</i>|/<i>τ</i></td>
      <td><i>C/K</i></td>
      <td>None</td>
      <td>None</td>
      <td>?</td>
      <td>BUND. ?</td>
    </tr>
    <tr>
      <td>Saito and Jun [2000]<sup>2</sup></td>
      <td>?</td>
      <td>?</td>
      <td>BS-?</td>
      <td>?</td>
      <td>?</td>
      <td>S&amp;P500. ?</td>
    </tr>
  </tbody>
</table>

<sup>7</sup>本文依據 Geigle [1999] 的博士論文。

6

<table>
  <thead>
    <tr>
      <th>作者 & 年份</th>
      <th>特徵</th>
      <th>輸出</th>
      <th>基準模型</th>
      <th>績效衡量指標</th>
      <th>資料分割方法</th>
      <th>標的資產</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>White [2000]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MAE, MSE</td>
      <td>隨機分割 (Random)</td>
      <td>模擬資料 (BS)；<br />Eurodollar. 7M</td>
    </tr>
    <tr>
      <td>Yao et al. [2000]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td><i>R</i><sup>2</sup></td>
      <td>時間順序分割 (Chronological)</td>
      <td>NIKKEI225. 1Y</td>
    </tr>
    <tr>
      <td>Dugas et al. [2001, 2009]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>無</td>
      <td>MSE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>S&amp;P500. 5Y</td>
    </tr>
    <tr>
      <td>Gençay and Qi [2001]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-H</td>
      <td>DM, <b>MATE</b>, MSE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>S&amp;P500. 6Y</td>
    </tr>
    <tr>
      <td>le Roux and du Toit [2001]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>無</td>
      <td>MSE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>模擬資料 (BS)</td>
    </tr>
    <tr>
      <td>Meissner and Kawano [2001]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>G</sub></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-G</td>
      <td>MAE, MAPE, ME, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>個別股票. 8M</td>
    </tr>
    <tr>
      <td>Schittenkopf and Dorffner [2001]</td>
      <td><i>τ</i></td>
      <td>高斯參數<sup>8</sup></td>
      <td>BS-H, CS</td>
      <td>MAE, <b>MATE</b>, ME, MSE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>FTSE100. 5Y</td>
    </tr>
    <tr>
      <td>Andreou et al. [2002]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>σ</i><sub>V</sub>, <i>r</i>, 及其他</td>
      <td><i>C</i>/<i>K</i>,<br />(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i>,<br />(<i>C</i> − <i>C</i><sub>BS−V</sub>)/<i>K</i></td>
      <td>BS-H, BS-V</td>
      <td>MdAE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>S&amp;P500. 3Y</td>
    </tr>
    <tr>
      <td>Billio et al. [2002]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>r</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-?</td>
      <td>MSE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>FTSE100. 1Y</td>
    </tr>
    <tr>
      <td>Ghosn and Bengio [2002]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>無</td>
      <td>MSE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>S&amp;P500. 6Y</td>
    </tr>
    <tr>
      <td>Healy et al. [2002]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>r</i>, spread, open interest, volume</td>
      <td><i>C</i></td>
      <td>無</td>
      <td>MAE, ME, <i>R</i><sup>2</sup></td>
      <td>隨機分割 (Random)</td>
      <td>FTSE100. 5Y</td>
    </tr>
    <tr>
      <td>Zapart [2002, 2003b]</td>
      <td>落後小波係數 (Lagged wavelet coefficients)</td>
      <td>小波係數<sup>9</sup></td>
      <td>BS-?</td>
      <td>MAE</td>
      <td>時間順序分割 (Chronological)</td>
      <td>個別股票. 6M/1Y</td>
    </tr>
    <tr>
----- END SOURCE -----

<table>
  <tbody>
    <tr>
      <td>Amilon [2003]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i>, 落後 <i>S</i></td>
      <td><i>C</i><sub>Ask</sub>/<i>K</i>,<br /><i>C</i><sub>Bid</sub>/<i>K</i></td>
      <td>BS-H, BS-IM</td>
      <td>ME, <b>MTE</b>, MSE</td>
      <td>時間順序</td>
      <td>OMX. 2Y</td>
    </tr>
    <tr>
      <td>Carverhill and Cheuk [2003]</td>
      <td><i>K</i>/<i>S</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>r</i></td>
      <td><i>C</i>/<i>K</i>, HR</td>
      <td>CRR</td>
      <td><b>?TE</b></td>
      <td>時間順序</td>
      <td>S&amp;P500. 11Y</td>
    </tr>
    <tr>
      <td>Gençay and Salih [2003]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-H</td>
      <td>DM, MSE</td>
      <td>時間順序</td>
      <td>S&amp;P500. 6Y</td>
    </tr>
    <tr>
      <td>Healy et al. [2003, 2004]<sup>10</sup></td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>無</td>
      <td>MSE, <i>R</i><sup>2</sup></td>
      <td>隨機</td>
      <td>FTSE100. 6Y</td>
    </tr>
    <tr>
      <td>Lajbcygier [2003, 2004]</td>
      <td><i>S</i>/<i>K</i>, <i>τ</i></td>
      <td>(<i>C</i> − <i>C</i><sub>BS−IH</sub>)/<i>K</i></td>
      <td>無</td>
      <td>MAE, MSE, <i>R</i><sup>2</sup></td>
      <td>時間順序</td>
      <td>AOSPI. 3Y</td>
    </tr>
    <tr>
      <td>Montagna et al. [2003]</td>
      <td><i>S</i>, <i>τ</i></td>
      <td><i>C</i></td>
      <td>無</td>
      <td>無</td>
      <td>?</td>
      <td>模擬 (BS)</td>
    </tr>
    <tr>
      <td>Zapart [2003a]<sup>11</sup></td>
      <td><i>S</i>/<i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i>/<i>K</i></td>
      <td>BS-?</td>
      <td>MAE</td>
      <td>時間順序</td>
      <td>個別股票. ?</td>
    </tr>
  </tbody>
</table>

<sup>8</sup> 人工神經網路（ANN）輸出高斯混合密度（Gaussian mixture density）的參數，以作為風險中性密度（risk-neutral density）的模型。

<sup>9</sup> 人工神經網路（ANN）被用來預測標的資產的未來波動率。波動率以小波（wavelets）表示，而標的資產則被建模為二項式樹（binomial tree）。

<sup>10</sup> 這些論文也推導了人工神經網路（ANN）期權價格估計值的預測區間（prediction intervals）。

<sup>11</sup> 這篇論文也處理了 Zapart [2002] 的設定。

<table>
  <thead>
    <tr>
      <th>作者 & 年份</th>
      <th>特徵</th>
      <th>輸出</th>
      <th>基準模型</th>
      <th>績效衡量指標</th>
      <th>資料分割方法</th>
      <th>標的資產</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Bennell and Sutcliffe [2004]</td>
      <td><i>S</i>, <i>K</i>, <i>S/K</i>, <i>τ</i>,<br /><i>σ</i><sub>IM</sub>, 未平倉量, 成交量</td>
      <td><i>C</i>, <i>C/K</i></td>
      <td>BS-IM</td>
      <td>MAE, ME, MPE, MSE</td>
      <td>Chronological</td>
      <td>FTSE100. 1Y</td>
    </tr>
    <tr>
      <td>Choi et al. [2004]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>?</sub></td>
      <td><i>C</i></td>
      <td>BS-?</td>
      <td>?</td>
      <td>Random</td>
      <td>KOSPI200. 1Y</td>
    </tr>
    <tr>
      <td>Dindar and Marwala [2004]</td>
      <td><i>S</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>K/C</i></td>
      <td>None</td>
      <td>?</td>
      <td>Random</td>
      <td>South Africa Foreign Exchange. 3Y</td>
    </tr>
    <tr>
      <td>Morelli et al. [2004]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>None</td>
      <td>?</td>
      <td>?</td>
      <td>模擬資料 (BS)</td>
    </tr>
    <tr>
      <td>Pires and Marwala [2004a,b]</td>
      <td><i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>SVM</td>
      <td>MAX, ME</td>
      <td>?</td>
      <td>Johannesburg Stock Exchange. 3Y</td>
    </tr>
    <tr>
      <td>Xu et al. [2004]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>I</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>None</td>
      <td><i>R</i><sup>2</sup></td>
      <td>Random</td>
      <td>FTSE100. 5Y</td>
    </tr>
    <tr>
      <td>Charalambous and Martzoukos [2005]</td>
      <td><i>S</i>, <i>K</i>, <i>σ</i><sub>H</sub>, <i>r</i>,<br />相關係數<sup>12</sup></td>
      <td><i>C</i> − <i>C</i><sub>LA</sub><sup>10</sup></td>
      <td>LA-10</td>
      <td>MAE, MAX, MSE</td>
      <td>Chronological</td>
      <td>模擬資料 (BS)</td>
    </tr>
    <tr>
      <td>Hamid and Habib [2005]</td>
      <td><i>S</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>None</td>
      <td>MAE, MSE</td>
      <td>?</td>
      <td>S&amp;P500. 12Y</td>
    </tr>
    <tr>
      <td>Kakati [2005]<sup>2</sup></td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>?</td>
      <td>個別股票. ?</td>
    </tr>
    <tr>
      <td>Ko et al. [2005], Ko [2009]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td>係數<sup>13</sup></td>
      <td>BS-H</td>
      <td><strong>MATE</strong></td>
      <td>?</td>
      <td>TAIEX. 1Y/2Y</td>
    </tr>
    <tr>
      <td>Lin and Yeh [2005]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MAE, MSE</td>
      <td>?</td>
      <td>TAIEX. 2Y</td>
    </tr>
    <tr>
      <td>Pires and Marwala [2005]</td>
      <td><i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>SVM</td>
      <td>MAX, ME, MSE</td>
      <td>?</td>
      <td>ALSI. 3Y</td>
    </tr>
    <tr>
      <td>Tung and Quek [2005]</td>
      <td><i>S</i> − <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>None</td>
      <td>MSE, Correlation<sup>14</sup></td>
      <td>?</td>
      <td>?</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>論文</th>
      <th>輸入變數</th>
      <th>輸出變數</th>
      <th>定價模型</th>
      <th>評估指標</th>
      <th>資料分割</th>
      <th>資料集</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Andreou et al. [2006]<sup>15</sup></td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>Cal</sub>,<br /><i>σ</i><sub>H</sub>, <i>σ</i><sub>V</sub>, <i>r</i></td>
      <td><i>C/K</i>, (<i>C</i> − <i>C</i><sub>BS−Cal</sub>)/<i>K</i>,<br />(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i>,<br />(<i>C</i> − <i>C</i><sub>BS−V</sub>)/<i>K</i></td>
      <td>BS-Cal, BS-H, BS-V</td>
      <td>MAE, MSE</td>
      <td>Chronological</td>
      <td>S&amp;P500. 3Y</td>
    </tr>
    <tr>
      <td>Blynski and Faseruk [2006]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>σ</i><sub>IH</sub></td>
      <td><i>C/K</i>,<br />(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i>,<br />(<i>C</i> − <i>C</i><sub>BS−N</sub>)/<i>K</i></td>
      <td>BS-H, BS-IH</td>
      <td>MAE, MAPE, ME, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>S&amp;P100. 7Y</td>
    </tr>
    <tr>
      <td>Huang and Wu [2006], Huang [2008]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>K</sub></td>
      <td>(<i>C</i> − <i>C</i><sub>BS−K</sub>)/<i>K</i></td>
      <td>SVM</td>
      <td>MAE, MAPE, MSE</td>
      <td>Chronological</td>
      <td>TAIEX. 9M</td>
    </tr>
    <tr>
      <td>Jung et al. [2006]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>IH</sub></td>
      <td>C</td>
      <td>BS-IH</td>
      <td>MSE</td>
      <td>?</td>
      <td>KOSPI200. 1Y</td>
    </tr>
    <tr>
      <td>Kim et al. [2006]</td>
      <td><i>K</i>, <i>τ</i></td>
      <td><i>σ</i><sub>Cal</sub></td>
      <td>SI</td>
      <td>MSE</td>
      <td>?</td>
      <td>S&amp;P500. 1M</td>
    </tr>
  </tbody>
</table>
<sup>12</sup>標的資產間的相關性。
<sup>13</sup>用以回歸期權價格的線性迴歸係數。
<sup>14</sup>皮爾森相關係數，一種用來驗證預測函數與目標函數配適度的統計指標。
<sup>15</sup>本文依據 Andreou [2008] 的博士論文。

<table>
<thead>
<tr>
<th>作者 & 年份</th>
<th>特徵</th>
<th>輸出</th>
<th>基準模型</th>
<th>績效衡量指標</th>
<th>資料分割方法</th>
<th>標的資產</th>
</tr>
</thead>
<tbody>
<tr>
<td>Liang et al. [2006]</td>
<td>$\hat{C}^{16}$</td>
<td>$C$</td>
<td>BS-?, CRR</td>
<td>MAE</td>
<td>Chronological</td>
<td>個別股票。5M</td>
</tr>
<tr>
<td>Mitra [2006]</td>
<td>$S, K, \tau, \sigma_{\text{H}}, r$</td>
<td>$C$</td>
<td>None</td>
<td>MAE, MSE</td>
<td>Chronological</td>
<td>NIFTY50。1Y</td>
</tr>
<tr>
<td>Pande and Sahu [2006]</td>
<td>$S/K, \tau, \sigma_{\text{PCA}}, r$</td>
<td>$C$ 或(?) $C/K$</td>
<td>None</td>
<td>ME, MSE, Correlation<sup>17</sup></td>
<td>?</td>
<td>個別股票。1Y</td>
</tr>
<tr>
<td>Teddy et al. [2006]</td>
<td>$S - K, \tau, \sigma_{\text{H}}$</td>
<td>$C$</td>
<td>None</td>
<td>MSE, Correlation<sup>14</sup></td>
<td>Random</td>
<td>GBP-USD。1Y</td>
</tr>
<tr>
<td>Tzastoudis et al. [2006]</td>
<td>$S, K, \sigma_{\text{H}}$</td>
<td>$C$</td>
<td>BS-H</td>
<td>MAE, $R^2$</td>
<td>Chronological</td>
<td>S&amp;P500。數天</td>
</tr>
<tr>
<td>Wang [2006]</td>
<td>$S/K, \sigma_{\text{IH}},$<br />$(S - K)^+,$<br />$C - (S - K)^+,$<br />$CS/\sqrt{K}$</td>
<td>$\sigma_{\text{I}}$</td>
<td>BS-IH</td>
<td>MAE, MSE, $R^2$</td>
<td>?</td>
<td>個別股票。2M</td>
</tr>
<tr>
<td>Amornwattana et al. [2007]</td>
<td>$S, K, \tau, r$</td>
<td>$C - C_{\text{BS-N}}, \sigma_{\text{I}}$</td>
<td>BS-H, BS-N</td>
<td>MAE, MSE</td>
<td>Chronological</td>
<td>個別股票。3M</td>
</tr>
<tr>
<td>Gençay and Gibson [2007]</td>
<td>$S, K, \tau, \sigma_{\text{G}}, r$</td>
<td>$C$</td>
<td>BS-G, BS-H, SV, SVJ</td>
<td>MAE, MSE</td>
<td>?</td>
<td>S&amp;P500。3Y</td>
</tr>
<tr>
<td>Gregoriou et al. [2007]</td>
<td>$S_{\text{Ask}}, S_{\text{Bid}},$<br />$S_{\text{Mid}}, K, \tau, \sigma_{\text{I}}, r$</td>
<td>$C$</td>
<td>None</td>
<td>None</td>
<td>Random</td>
<td>FTSE100。5Y</td>
</tr>
<tr>
<td>Healy et al. [2007]</td>
<td>$S, K, \tau, \sigma_{\text{I}}, r$</td>
<td>$C$</td>
<td>None</td>
<td>$R^2$</td>
<td>Chronological</td>
<td>FTSE100。?</td>
</tr>
<tr>
<td>Thomaidis et al. [2007]</td>
<td>$S, K, \tau$</td>
<td>$C$</td>
<td>BS-G, BS-H</td>
<td>MAE, MSE</td>
<td>Chronological</td>
<td>S&amp;P500。數天</td>
</tr>
<tr>
<td>Zhou et al. [2007]</td>
<td>$S/K, S, K, \tau, r$</td>
<td>$C/K$</td>
<td>BS-?, CRR</td>
<td>MAE, MAPE, ME, MSE, $R^2$</td>
<td>Chronological</td>
<td>可轉換公司債。2Y</td>
</tr>
<tr>
<td>Andreou et al. [2008]<sup>15</sup></td>
<td>$S/K, \sigma_{\text{Cal}}, \sigma_{\text{H}},$<br />$\sigma_{\text{V}}, r,$ kurtosis,<br />skewness</td>
<td>$C/K, (C - C_{\text{BS-Cal}})/K,$<br />$(C - C_{\text{BS-H}})/K,$<br />$(C - C_{\text{BS-V}})/K$</td>
<td>BS-Cal, BS-H, BS-V, CS</td>
<td>MAE, <b>MATE</b>, MdAE, MSE, <b>MTE</b></td>
<td>Chronological</td>
<td>S&amp;P500。4Y</td>
</tr>
<tr>
<td>Chiu and Lin [2008]</td>
<td>$S, C_{\text{BS}}$, volume, and others</td>
<td>$C$</td>
<td>None</td>
<td>MSE</td>
<td>Chronological</td>
<td>個別股票。1Y</td>
</tr>
<tr>
<td>Kakati [2008]</td>
<td>$S/K, \tau, \sigma_{\text{G}}, \sigma_{\text{H}},$<br />$\sigma_{\text{IH}}, r$</td>
<td>$C/K$</td>
<td>BS-G, BS-H, BS-IH</td>
<td>MSE</td>
<td>?</td>
<td>個別股票。數天</td>
</tr>
<tr>
<td>Mostafa and Dillon [2008]<sup>18</sup></td>
<td>$S/K, \tau, \sigma_{\text{H}}$</td>
<td>$C/K, \sigma_{\text{I}}$</td>
<td>BS-H, SV</td>
<td>MAPE, <b>MATE</b>, MPE</td>
<td>?</td>
<td>FTSE100。2Y</td>
</tr>
</tbody>
</table>

***

<sup>16</sup> 各種參數化選擇權定價模型所估計的價格。

<sup>17</sup> 實際價格與計算價格之間的相關性。

<sup>18</sup> 本文依據 Mostafa [2011] 的博士論文。

<table>
  <thead>
    <tr>
      <th>作者 & 年份</th>
      <th>特徵</th>
      <th>輸出</th>
      <th>基準模型</th>
      <th>績效衡量指標</th>
      <th>資料分割方法</th>
      <th>標的資產</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Quek et al. [2008]</td>
      <td>lagged <i>C</i></td>
      <td><i>C</i></td>
      <td>無</td>
      <td>無</td>
      <td>?</td>
      <td>GBP-USD、黃金、石油。2 年</td>
    </tr>
    <tr>
      <td>Saxena [2008]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td>(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i></td>
      <td>BS-H</td>
      <td>MAE, ME, MPE, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>NIFTY50。1 年</td>
    </tr>
    <tr>
      <td>Teddy et al. [2008]</td>
      <td><i>S</i> − <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MSE, <i>R</i><sup>2</sup></td>
      <td>隨機</td>
      <td>GBP-USD。9 個月</td>
    </tr>
    <tr>
      <td>Tseng et al. [2008]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>G</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>無</td>
      <td>MAE, MAPE, MSE</td>
      <td>?</td>
      <td>TAIEX。2 年</td>
    </tr>
    <tr>
      <td>Chen [2009]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>BS-H, SVM</td>
      <td>MAE, MSE</td>
      <td>時間順序</td>
      <td>S&amp;P500。數天</td>
    </tr>
    <tr>
      <td>Gradojevic et al. [2009]</td>
      <td><i>S/K</i>, <i>τ</i></td>
      <td><i>C/K</i></td>
      <td>BS-H</td>
      <td>DM, MSE, MSPE</td>
      <td>時間順序</td>
      <td>S&amp;P500。8 年</td>
    </tr>
    <tr>
      <td>Leung et al. [2009]</td>
      <td><i>σ</i><sub>H</sub>, <i>σ</i><sub>IH</sub>, 成交量, 未平倉量</td>
      <td><i>σ</i><sub>I</sub></td>
      <td>BS-IH, 線性, 多項式</td>
      <td>ME</td>
      <td>時間順序</td>
      <td>多種貨幣。17 年</td>
    </tr>
    <tr>
      <td>Liang et al. [2009]</td>
      <td><i>Ĉ</i><sup>16</sup></td>
      <td><i>C</i></td>
      <td>CRR, SVM</td>
      <td>MAE, MAPE</td>
      <td>時間順序</td>
      <td>個別股票。2 年</td>
    </tr>
    <tr>
      <td>Martel et al. [2009]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i><sub>Bid</sub>/<i>K</i>,<br /><i>C</i><sub>Ask</sub>/<i>K</i></td>
      <td>BS-H</td>
      <td>ME, MSE, <b>MTE</b></td>
      <td>時間順序</td>
      <td>IBEX35。2 年</td>
    </tr>
    <tr>
      <td>Samur and Temur [2009]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>, <i>r</i></td>
      <td><i>C</i></td>
      <td>無</td>
      <td>MAE, MSE, <i>R</i><sup>2</sup></td>
      <td>?</td>
      <td>S&amp;P100。數天</td>
    </tr>
    <tr>
      <td>Wang [2009a]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>G</sub>, <i>r</i></td>
      <td><i>C/K</i></td>
      <td>無</td>
      <td>MAE, MAPE, MSE</td>
      <td>?</td>
      <td>TAIEX。2 年</td>
    </tr>
    <tr>
      <td>Wang [2009b]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>G</sub>, <i>σ</i><sub>H</sub>,<br /><i>σ</i><sub>IH</sub>, <i>r</i></td>
      <td><i>C/K</i></td>
      <td>無</td>
      <td>MAE, MAPE, MSE</td>
      <td>?</td>
      <td>TAIEX。2 年</td>
    </tr>
    <tr>
      <td>Andreou et al. [2010]<sup>15</sup></td>
      <td><i>S/K</i>, <i>τ</i></td>
    </tr>
  </tbody>
</table>

<td><i>σ</i><sub>I</sub></td>
      <td>BS-Cal, CS, SV, SVJ</td>
      <td>MAE, <b>MATE</b>, MdAE, MSE</td>
      <td>Chronological</td>
      <td>S&#x26;P500. 3Y</td>
    </tr>
    <tr>
      <td>Barunikova and Barunik [2011]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i></td>
      <td><i>C</i></td>
      <td>BS-H</td>
      <td>MAE, MAPE, MSE</td>
      <td>Random</td>
      <td>S&#x26;P500. 3Y</td>
    </tr>
    <tr>
      <td>Gradojevic and Kukolj [2011]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>IH</sub>, <i>r</i></td>
      <td><i>C/K</i></td>
      <td>BS-H</td>
      <td>DM, MAPE, MSE</td>
      <td>Chronological</td>
      <td>S&#x26;P500. 7Y</td>
    </tr>
    <tr>
      <td>Liu and Zhang [2011]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>H</sub>,<sup>19</sup> <i>r</i></td>
      <td><i>C/K</i></td>
      <td>BS-H</td>
      <td>MAE, MSE</td>
      <td>Chronological</td>
      <td>Individual stocks. 2Y</td>
    </tr>
    <tr>
      <td>Phani et al. [2011]</td>
      <td><i>S</i>, <i>K</i>, <i>τ</i></td>
      <td><i>C</i></td>
      <td>BS-?, SVM</td>
      <td>MAE</td>
      <td>?</td>
      <td>NIFTY50. 2Y</td>
    </tr>
    <tr>
      <td>Tung and Quek [2011]</td>
      <td><i>σ</i><sub>IH</sub></td>
      <td><i>σ</i><sub>I</sub></td>
      <td>None</td>
      <td>MAPE, MSE, <i>R</i><sup>2</sup></td>
      <td>Chronological</td>
      <td>HSI. 5Y</td>
    </tr>
    <tr>
      <td>Wang [2011]</td>
      <td><i>S/K</i>, <i>S</i>, <i>τ</i>, <i>σ</i><sub>Cal</sub>,<br /><i>r</i></td>
      <td><i>C</i></td>
      <td>SV, SVJ, SVM</td>
      <td>MAE, MAPE</td>
      <td>Chronological</td>
      <td>Several currencies. 7M</td>
    </tr>
  </tbody>
</table>

<sup>19</sup>更精確地說，此處使用馬可夫體制轉換模型（Markov regime switching model）來估計波動率。

9

10

<table>
<thead>
<tr>
<th>作者 & 年份</th>
<th>特徵</th>
<th>輸出</th>
<th>基準模型</th>
<th>績效衡量指標</th>
<th>資料分割方法</th>
<th>標的資產</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ahn et al. [2012]</td>
<td>落後 σ<sub>I</sub>、<br />Greeks</td>
<td>Sign(Δσ<sub>I</sub>)</td>
<td>無</td>
<td>準確率</td>
<td>時間順序</td>
<td>KOSPI200，2 年</td>
</tr>
<tr>
<td>Chen and Sutcliffe [2012]</td>
<td><i>S/K</i>、<i>τ</i></td>
<td><i>C/K</i>、<br />(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i>、<br />HR</td>
<td>BS-H</td>
<td>MAE、ME、MSE</td>
<td>隨機</td>
<td>Sterling 期貨，2 年</td>
</tr>
<tr>
<td>Mitra [2012]</td>
<td><i>S</i>、<i>K</i>、<i>τ</i>、σ<sub>H</sub>、<i>r</i></td>
<td><i>C</i></td>
<td>BS-H</td>
<td>ME、MSE</td>
<td>時間順序</td>
<td>NIFTY50，3 年</td>
</tr>
<tr>
<td>Shin and Ryu [2012]</td>
<td><i>S</i>、<i>K</i>、<i>τ</i>、<i>r</i></td>
<td>HR</td>
<td>無</td>
<td>MPE</td>
<td>時間順序</td>
<td>KOSPI200，10 年</td>
</tr>
<tr>
<td>Wang et al. [2012]</td>
<td><i>S</i>、<i>K</i>、<i>τ</i>、σ<sub>Cal</sub>、<br />σ<sub>G</sub>、σ<sub>H</sub>、σ<sub>IH</sub></td>
<td><i>C</i></td>
<td>無</td>
<td>MAE、MAPE、<br />MSE</td>
<td>時間順序</td>
<td>TAIEX，2 年</td>
</tr>
<tr>
<td>Chang et al. [2013]</td>
<td><i>S/K</i>、<i>τ</i>、σ<sub>G</sub>、<i>r</i></td>
<td><i>C</i> 或 <i>C/K</i></td>
<td>無</td>
<td>MAE、MAPE</td>
<td>?</td>
<td>TAIEX，2 年</td>
</tr>
<tr>
<td>Hahn [2013]</td>
<td><i>S/K</i>、<i>τ</i>、σ<sub>H</sub>、<i>r</i></td>
<td><i>C/K</i></td>
<td>SV</td>
<td>MAE、MAPE、<br />MSE</td>
<td>時間順序</td>
<td>個別股票，10 年</td>
</tr>
<tr>
<td>Can and Fadda [2014]</td>
<td><i>S/K</i>、<i>S</i>、<i>τ</i>、<i>r</i></td>
<td><i>C/K</i></td>
<td>BS-H</td>
<td>MAE</td>
<td>時間順序</td>
<td>S&amp;P100，數日</td>
</tr>
<tr>
<td>Lai [2014]</td>
<td><i>S/K</i>、<i>τ</i>、<i>r</i></td>
<td>σ<sub>I</sub></td>
<td>KR、SI</td>
<td>KS</td>
<td>?</td>
<td>模擬 (BS、SV、SVJ)</td>
</tr>
<tr>
<td>Park et al. [2014]</td>
<td><i>S/K</i>、<i>τ</i></td>
<td><i>C/K</i></td>
<td>BS-H、SV</td>
<td>MSE</td>
<td>時間順序</td>
<td>KOSPI200，10 年</td>
</tr>
<tr>
<td>von Spreckelsen et al. [2014]</td>
<td><i>S/K</i>、<i>K</i>、<i>τ</i></td>
<td><i>C/K</i></td>
<td>無</td>
<td>MSE、<i>R</i><sup>2</sup></td>
<td>時間順序</td>
<td>EUR-USD，1 個月</td>
</tr>
<tr>
<td>Ludwig [2015]</td>
<td><i>S/K</i>、<i>τ</i></td>
<td>σ<sub>I</sub></td>
<td>Quadratic</td>
<td>MSE、<i>R</i><sup>2</sup></td>
<td>?</td>
<td>S&amp;P500，12 年</td>
</tr>
<tr>
<td>Liu and Huang [2016]</td>
<td><i>S/K</i>、<i>τ</i></td>
<td>(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i></td>
<td>BS-H</td>
<td>MAE、MAPE、<br />ME、MSE</td>
<td>?</td>
<td>HSI，6 年</td>
</tr>
<tr>
<td>Montesdeoca and Niranjan [2016]</td>
<td><i>S/K</i>、<i>τ</i>、σ<sub>H</sub>、<br />成交量</td>
<td><i>C/K</i></td>
<td>無</td>
<td>MSE</td>
<td>時間順序</td>
<td>FTSE100，?；<br />個別股票，?</td>
</tr>
<tr>
<td>Culkin and Das [2017]</td>
<td><i>S/K</i>、<i>τ</i>、σ<sub>I</sub>、<i>r</i></td>
<td><i>C/K</i></td>
<td>無</td>
<td>MSE、<i>R</i><sup>2</sup></td>
<td>時間順序</td>
<td>模擬 (BS)</td>
</tr>
<tr>
<td>Das and Padhy [2017]</td>
<td><i>S/K</i>、<i>τ</i>、<i>Ĉ</i><sup>16</sup></td>
<td><i>C</i></td>
<td>BS-H、SVM</td>
<td>MAE、MSE</td>
<td>時間順序</td>
<td>NIFTY50，2 年</td>
</tr>
<tr>
<td>Fang and George [2017]</td>
<td>σ<sub>H</sub></td>
<td>σ<sub>I</sub></td>
<td>無</td>
<td>MSE、<i>R</i><sup>2</sup></td>
<td>時間順序</td>
<td>模擬 (BS)；WTI，1 個月</td>
</tr>
<tr>
<td>Palmer and Gorse [2017]</td>
<td><i>S</i>、<i>K</i>、σ<sub>I</sub>、<i>r</i></td>
<td><i>C</i></td>
<td>無</td>
<td>MAE、MdAE、<br />MAPE</td>
<td>時間順序</td>
<td>模擬 (BS)</td>
</tr>
<tr>
<td>Yang et al. [2017]<sup>20</sup></td>
<td><i>K/S</i>、<i>τ</i></td>
<td><i>C/S</i></td>
<td>BS-?、Kou、VG</td>
<td>MAPE、MSE</td>
<td>?</td>
<td>S&amp;P500，10 年</td>
</tr>
<tr>
<td>Ferguson and Green [2018]</td>
<td><i>S</i>、<i>τ</i>、σ<sub>I</sub>、<br />相關係數<sup>12</sup></td>
<td><i>C</i></td>
<td>無</td>
<td>MSE</td>
<td>時間順序</td>
<td>模擬 (BS)</td>
</tr>
<tr>
<td>Ackerer et al. [2019]</td>
<td>log(<i>K/S</i>)、<i>τ</i>、<br />log(<i>K/S</i>)<i>τ</i><sup>−0.5</sup>、<br />log(<i>K/S</i>)<i>τ</i><sup>−0.95</sup></td>
<td>σ<sub>I</sub></td>
<td>無</td>
<td>MAPE、MSE</td>
<td>隨機</td>
<td>S&amp;P500，1 個月</td>
</tr>
</tbody>
</table>

<sup>20</sup>本文依據鄭（Zheng）[2017] 的博士論文。

<table>
  <thead>
    <tr>
      <th>作者 & 年份</th>
      <th>特徵</th>
      <th>輸出</th>
      <th>基準模型</th>
      <th>績效衡量指標</th>
      <th>資料分割方法</th>
      <th>標的資產</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Buehler et al. [2019a,b]</td>
      <td>log(<i>S</i>)</td>
      <td>HR</td>
      <td>BS-I</td>
      <td><strong>CVaR</strong></td>
      <td>Chronological</td>
      <td>模擬 (BS, SV)；<br />S&amp;P500. 5Y</td>
    </tr>
    <tr>
      <td>Cao et al. [2019]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>V</sub>,<br />underlying return</td>
      <td><i>σ</i><sub>I</sub></td>
      <td>HW</td>
      <td>MSE</td>
      <td>Random</td>
      <td>S&amp;P500. 8Y</td>
    </tr>
    <tr>
      <td>Jang and Lee [2019]</td>
      <td>?</td>
      <td><i>C</i></td>
      <td>BS-Cal, BW,<br />KR, LSM, LV,<br />SVJ, SVM</td>
      <td>MAE, MAPE,<br />MPE, MSE</td>
      <td>?</td>
      <td>S&amp;P100. 9Y</td>
    </tr>
    <tr>
      <td>Liu et al. [2019b]</td>
      <td><i>S/K</i>, <i>τ</i></td>
      <td><i>σ</i><sub>I</sub></td>
      <td>None</td>
      <td>MAE, MAPE,<br />MSE</td>
      <td>Chronological</td>
      <td>模擬 (BS)</td>
    </tr>
    <tr>
      <td>Liu et al. [2019c]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>σ</i><sub>Cal</sub>, <i>r</i></td>
      <td>(<i>C</i> − <i>C</i><sub>BS−H</sub>)/<i>K</i></td>
      <td>BS-Cal, SVJ</td>
      <td>MAE, <strong>MATE</strong>,<br />MPE, MSE</td>
      <td>Chronological</td>
      <td>DAX. 4Y</td>
    </tr>
    <tr>
      <td>Karatas et al. [2019]</td>
      <td><i>S/K</i>, <i>τ</i>, <i>r</i>, ?</td>
      <td><i>C/K</i></td>
      <td>None</td>
      <td>MSE, <i>R</i><sup>2</sup></td>
      <td>Chronological</td>
      <td>模擬 (BS, SV, VG)</td>
    </tr>
    <tr>
      <td>Palmer [2019]</td>
      <td><i>S/K</i>, <i>σ</i><sub>I</sub>√<i>τ</i>, <i>r</i></td>
      <td><i>C/K</i></td>
      <td>BS-I, LSM</td>
      <td>MAE, MAPE</td>
      <td>Chronological</td>
      <td>模擬 (BS)</td>
    </tr>
    <tr>
      <td>Zheng et al. [2019]</td>
      <td><i>S/K</i>, <i>τ</i></td>
      <td><i>σ</i><sub>I</sub></td>
      <td>SSVI</td>
      <td>MAPE</td>
      <td>?</td>
      <td>S&amp;P500. 10Y</td>
    </tr>
    <tr>
      <td>Ruf and Wang [2020]</td>
      <td><i>S/K</i>, <i>σ</i><sub>I</sub>√<i>τ</i>, Δ,<br />𝒱, Vanna</td>
      <td>HR</td>
      <td>BS-I, HW,<br />Linear</td>
      <td>MSE</td>
      <td>Chronological</td>
      <td>模擬 (BS, SV)；<br />S&amp;P500. 8Y；<br />STOXX50. 3Y</td>
    </tr>
  </tbody>
</table>

表 1：此表格彙總了超過 150 篇將人工神經網路（ANN）作為非參數選擇權定價或避險工具的論文。這些論文在特徵（或稱解釋變數）、人工神經網路的輸出、基準模型、訓練與測試集的資料分割方式，以及標的資產與資料涵蓋的時間範圍等方面進行比較。粗體標示的績效衡量指標與多期間評估相關。所有縮寫的對照請參見表 2 至表 5。

<table>
<tr>
<td><i>C</i></td>
<td>選擇權價格</td>
</tr>
<tr>
<td><i>C</i><sub>BS−X</sub></td>
<td>Black-Scholes 公式所給出的選擇權價格；X 的不同意義請見表 3</td>
</tr>
<tr>
<td><i>C</i><sub>LA</sub><sup><i>n</i></sup></td>
<td><i>n</i> 步多維格狀架構所給出的選擇權價格</td>
</tr>
<tr>
<td>HR</td>
<td>避險比率</td>
</tr>
<tr>
<td><i>K</i></td>
<td>履約價格</td>
</tr>
<tr>
<td><i>S</i></td>
<td>股票價格</td>
</tr>
<tr>
<td><i>r</i></td>
<td>利率</td>
</tr>
<tr>
<td>Γ</td>
<td>Gamma：選擇權價格對標的物價格的二階敏感度</td>
</tr>
<tr>
<td>Δ</td>
<td>Delta：選擇權價格對標的物價格的敏感度</td>
</tr>
<tr>
<td>𝒱</td>
<td>Vega：選擇權價格對波動率的敏感度</td>
</tr>
<tr>
<td>ρ</td>
<td>Rho：選擇權價格對利率的敏感度</td>
</tr>
<tr>
<td>σ<sub>Cal</sub></td>
<td>由校準所得之波動率（例如，在不同履約價與到期日間為常數）</td>
</tr>
<tr>
<td>σ<sub>G</sub></td>
<td>GARCH 所產生的波動率</td>
</tr>
<tr>
<td>σ<sub>H</sub></td>
<td>歷史波動率</td>
</tr>
<tr>
<td>σ<sub>I</sub></td>
<td>隱含波動率 (implied volatility)</td>
</tr>
<tr>
<td>σ<sub>IH</sub></td>
<td>隱含歷史波動率 (implied historical volatility)</td>
</tr>
<tr>
<td>σ<sub>IM</sub></td>
<td>價平隱含波動率 (at-the-money implied volatility)</td>
</tr>
<tr>
<td>σ<sub>K</sub></td>
<td>由卡爾曼濾波器 (Kalman filter) 所得之波動率</td>
</tr>
<tr>
<td>σ<sub>PCA</sub></td>
<td>經由主成分分析 (principal component analysis) 所決定、對波動率貢獻最大的總體經濟變數</td>
</tr>
<tr>
<td>σ<sub>V</sub></td>
<td>波動率指數，例如 VIX 與 DVAX</td>
</tr>
<tr>
<td>τ</td>
<td>距到期時間</td>
</tr>
</table>

表 2：此表呈現表 1 所使用的特徵與輸出之符號及縮寫。

<table>
    <tr>
      <td>BS-Cal</td>
      <td>Black-Scholes formula with calibrated volatility</td>
    </tr>
    <tr>
      <td>BS-G</td>
      <td>Black-Scholes formula with GARCH-generated volatility</td>
    </tr>
    <tr>
      <td>BS-H</td>
      <td>Black-Scholes formula with historical volatility</td>
    </tr>
    <tr>
      <td>BS-I</td>
      <td>Black-Scholes formula with contract-specific implied volatility</td>
    </tr>
    <tr>
      <td>BS-IH</td>
      <td>Black-Scholes formula with historical implied volatility</td>
    </tr>
    <tr>
      <td>BS-IM</td>
      <td>Black-Scholes formula with at-the-money implied volatility</td>
    </tr>
    <tr>
      <td>BS-K</td>
      <td>Black-Scholes formula with volatility obtained from Kalman filter</td>
    </tr>
    <tr>
      <td>BS-N</td>
      <td>Black-Scholes formula with ANN-generated volatility</td>
    </tr>
    <tr>
      <td>BS-V</td>
      <td>Black-Scholes formula with volatility index, such as VIX or VDAX</td>
    </tr>
    <tr>
      <td>BW</td>
      <td>Barone-Adesi and Whaley [1987] pricing method</td>
    </tr>
    <tr>
      <td>CRR</td>
      <td>Cox et al. [1979] model</td>
    </tr>
    <tr>
      <td>CS</td>
      <td>Corrado and Su [1996] model</td>
    </tr>
    <tr>
      <td>HW</td>
      <td>Hull and White [2017] model</td>
    </tr>
    <tr>
      <td>Kou</td>
      <td>Kou [2002]’s jump diffusion model</td>
    </tr>
    <tr>
      <td>KR</td>
      <td>Kernel regression</td>
    </tr>
    <tr>
      <td>LA-n</td>
      <td>n-step multi-dimensional lattice scheme</td>
    </tr>
    <tr>
      <td>Linear</td>
      <td>Linear regression on features</td>
    </tr>
    <tr>
      <td>LSM</td>
      <td>Longstaff and Schwartz [2001] method</td>
    </tr>
    <tr>
      <td>LV</td>
      <td>Local volatility model</td>
    </tr>
    <tr>
      <td>Quadratic</td>
      <td>Quadratic regression on features</td>
    </tr>
    <tr>
      <td>SI</td>
      <td>Spline interpolation</td>
    </tr>
    <tr>
      <td>SSVI</td>
      <td>Surface stochastic volatility inspired model, see Gatheral and Jacquier<br />[2014]</td>
    </tr>
    <tr>
      <td>SV</td>
      <td>Stochastic volatility models, such as Heston [1993] or GARCH</td>
    </tr>
    <tr>
      <td>SVJ</td>
      <td>Stochastic volatility with jumps model, see Bates [1996] or Carr et al.<br />[2003]</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>Support vector machine</td>
    </tr>
    <tr>
      <td>VG</td>
      <td>Variance Gamma model, see Madan et al. [1998]</td>
    </tr>
</table>

表 3：此表列出各種基準模型的縮寫，這些縮寫用於表 1。

<table>
<tr>
<td>DM</td>
<td>Diebold-Mariano 檢定 (Diebold and Mariano test)</td>
<td></td>
</tr>
<tr>
<td>KS</td>
<td>Kolmogorov-Smirnov 雙樣本檢定 (Kolmogorov and Smirnov two-sample test)</td>
<td></td>
</tr>
<tr>
<td>MAE</td>
<td>平均絕對誤差 (Mean absolute error)</td>
<td>$\frac{1}{N} \sum |\hat{y}_i - y_i|$</td>
</tr>
<tr>
<td>MAPE</td>
<td>平均絕對百分比誤差 (Mean absolute percentage error)</td>
<td>$\frac{1}{N} \sum \frac{|\hat{y}_i - y_i|}{y_i}$</td>
</tr>
<tr>
<td>MAX</td>
<td>最大誤差 (Maximum error)</td>
<td>$\max_i |\hat{y}_i - y_i|$</td>
</tr>
<tr>
<td>MdAE</td>
<td>中位數絕對誤差 (Median absolute error)</td>
<td>$\sup_z \left\{ \frac{1}{N} \sum \mathbf{1}_{|\hat{y}_i - y_i| < z} \leq 0.5 \right\}$</td>
</tr>
<tr>
<td>ME</td>
<td>平均誤差 (Mean error)</td>
<td>$\frac{1}{N} \sum (\hat{y}_i - y_i)$</td>
</tr>
<tr>
<td>MPE</td>
<td>平均百分比誤差 (Mean percentage error)</td>
<td>$\frac{1}{N} \sum \frac{\hat{y}_i - y_i}{y_i}$</td>
</tr>
<tr>
<td>MSE</td>
<td>均方誤差 (Mean squared error)</td>
<td>$\frac{1}{N} \sum (\hat{y}_i - y_i)^2$</td>
</tr>
<tr>
<td>$R^2$</td>
<td>決定係數 (Coefficient of determination)</td>
<td>$1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (\bar{y} - y_i)^2}$</td>
</tr>
<tr>
<td>SR</td>
<td>交易策略之夏普比率 (Sharpe ratio of a trading ratio)</td>
<td></td>
</tr>
<tr>
<td>%E</td>
<td>樣本百分比誤差 (Sample-wise percentage error)</td>
<td>$\frac{\hat{y}_i - y_i}{y_i}$</td>
</tr>
<tr>
<td><strong>CVaR</strong></td>
<td>條件風險價值 (Conditional value-at-risk)</td>
<td></td>
</tr>
<tr>
<td><strong>MATE</strong></td>
<td>平均絕對追蹤誤差 (Mean absolute tracking error)</td>
<td>$\frac{1}{N} \sum e^{-rT_i} |V(T_i)|$</td>
</tr>
<tr>
<td><strong>MTE</strong></td>
<td>平均追蹤誤差 (Mean tracking error)</td>
<td>$\frac{1}{N} \sum e^{-rT_i} V(T_i)$</td>
</tr>
<tr>
<td><strong>PE</strong></td>
<td>預測誤差 (Prediction error)</td>
<td>$\sqrt{\text{MTE}^2 + \frac{1}{N} \sum (e^{-rT_i} V(T_i) - \text{MTE})^2}$</td>
</tr>
</table>

**Table 4：此表呈現用於表 1 的績效衡量指標之縮寫與定義。其中，$\hat{y}_i$ 為估計的選擇權價格 / 隱含波動率 / 投資組合價值，$y_i$ 為目標值，$\bar{y}$ 為目標值的平均值，$N$ 表示樣本數。此外，$V(T)$ 又稱為追蹤誤差（tracking error），表示從零財富開始的避險選擇權投資組合在到期日 $T$ 的終值。所有以粗體標示的績效衡量指標均與多期評估相關。**

<table>
<caption>表 1：績效衡量指標</caption>
<thead>
<tr>
<th>縮寫</th>
<th>定義</th>
<th>數學公式</th>
</tr>
</thead>
<tbody>
<tr>
<td>MAE</td>
<td>平均絕對誤差 (Mean Absolute Error)</td>
<td>$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$</td>
</tr>
<tr>
<td><b>MAE</b></td>
<td><b>平均絕對誤差 (Mean Absolute Error)</b></td>
<td>$\frac{1}{n \cdot T} \sum_{t=1}^{T} \sum_{i=1}^{n} |y_{i,t} - \hat{y}_{i,t}|$</td>
</tr>
<tr>
<td>MSE</td>
<td>平均平方誤差 (Mean Squared Error)</td>
<td>$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$</td>
</tr>
<tr>
<td><b>MSE</b></td>
<td><b>平均平方誤差 (Mean Squared Error)</b></td>
<td>$\frac{1}{n \cdot T} \sum_{t=1}^{T} \sum_{i=1}^{n} (y_{i,t} - \hat{y}_{i,t})^2$</td>
</tr>
<tr>
<td>RMSE</td>
<td>均方根誤差 (Root Mean Squared Error)</td>
<td>$\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$</td>
</tr>
<tr>
<td><b>RMSE</b></td>
<td><b>均方根誤差 (Root Mean Squared Error)</b></td>
<td>$\sqrt{\frac{1}{n \cdot T} \sum_{t=1}^{T} \sum_{i=1}^{n} (y_{i,t} - \hat{y}_{i,t})^2}$</td>
</tr>
<tr>
<td>MAPE</td>
<td>平均絕對百分比誤差 (Mean Absolute Percentage Error)</td>
<td>$\frac{1}{n} \sum_{i=1}^{n} |\frac{y_i - \hat{y}_i}{y_i}|$</td>
</tr>
<tr>
<td><b>MAPE</b></td>
<td><b>平均絕對百分比誤差 (Mean Absolute Percentage Error)</b></td>
<td>$\frac{1}{n \cdot T} \sum_{t=1}^{T} \sum_{i=1}^{n} |\frac{y_{i,t} - \hat{y}_{i,t}}{y_{i,t}}|$</td>
</tr>
<tr>
<td>sMAPE</td>
<td>對稱平均絕對百分比誤差 (Symmetric Mean Absolute Percentage Error)</td>
<td>$\frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$</td>
</tr>
<tr>
<td><b>sMAPE</b></td>
<td><b>對稱平均絕對百分比誤差 (Symmetric Mean Absolute Percentage Error)</b></td>
<td>$\frac{1}{n \cdot T} \sum_{t=1}^{T} \sum_{i=1}^{n} \frac{|y_{i,t} - \hat{y}_{i,t}|}{(|y_{i,t}| + |\hat{y}_{i,t}|)/2}$</td>
</tr>
<tr>
<td>$R^2$</td>
<td>判定係數 (Coefficient of Determination)</td>
<td>$1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$</td>
</tr>
<tr>
<td><b>$R^2$</b></td>
<td><b>判定係數 (Coefficient of Determination)</b></td>
<td>$1 - \frac{\sum_{t=1}^{T} \sum_{i=1}^{n} (y_{i,t} - \hat{y}_{i,t})^2}{\sum_{t=1}^{T} \sum_{i=1}^{n} (y_{i,t} - \bar{y})^2}$</td>
</tr>
</tbody>
</table>

<table>
    <tr>
      <td>ALSI</td>
      <td>南非所有股票指數</td>
    </tr>
    <tr>
      <td>AOSPI</td>
      <td>澳洲所有普通股股價指數</td>
    </tr>
    <tr>
      <td>BUND</td>
      <td>德國公債</td>
    </tr>
    <tr>
      <td>DAX</td>
      <td>德國股票指數</td>
    </tr>
    <tr>
      <td>DEM</td>
      <td>德國馬克</td>
    </tr>
    <tr>
      <td>FTSE100</td>
      <td>英國金融時報股票交易所100指數</td>
    </tr>
    <tr>
      <td>HSI</td>
      <td>香港恆生指數</td>
    </tr>
    <tr>
      <td>IBEX35</td>
      <td>西班牙股票指數</td>
    </tr>
    <tr>
      <td>KOSPI200</td>
      <td>韓國綜合股價指數</td>
    </tr>
    <tr>
      <td>NIFTY50</td>
      <td>印度國家證券交易所五十指數</td>
    </tr>
    <tr>
      <td>NIKKEI225</td>
      <td>日本股票指數</td>
    </tr>
    <tr>
      <td>OMX</td>
      <td>瑞典股票指數</td>
    </tr>
    <tr>
      <td>S&#x26;P100</td>
      <td>美國標準普爾100指數</td>
    </tr>
    <tr>
      <td>S&#x26;P500</td>
      <td>美國標準普爾500指數</td>
    </tr>
    <tr>
      <td>STOXX50</td>
      <td>歐元區股票指數</td>
    </tr>
    <tr>
      <td>TAIEX</td>
      <td>台灣股票指數</td>
    </tr>
    <tr>
      <td>WTI</td>
      <td>美國輕質低硫原油期貨</td>
    </tr>
</table>
表 5：此表列出各種股票市場指數及其他標的資產的縮寫，這些縮寫用於表 1。關於描述模擬資料所使用的簡稱，請參見表 3。

以下我們根據特徵、輸出、績效衡量指標與基準、資料分割方法、標的資產及時間範圍，對表 1 所列論文進行比較與分類。

## 2.1 特徵

要估計選擇權價格，標的資產價格與履約價格是兩個不可或缺的變數。將這兩個變數輸入人工神經網路（ANN）的方式有兩種。一種是分別使用標的資產價格與履約價格；另一種替代方法則是使用其比率（即價內程度，moneyness）。文獻中提出幾項支持使用價內程度的論點：

*   使用價內程度而非分別使用股票價格與履約價格，可減少輸入變數的數量，從而使人工神經網路的訓練更為容易；見 Hutchinson et al. [1994]。
*   許多參數模型假設標的資產報酬率的統計分配與標的資產水準無關。因此，選擇權定價函數對於標的股票價格與履約價格而言是一階齊次函數（homogeneous of degree one），故僅需價內程度即可學習該函數。將此假設納入人工神經網路可潛在減少過度配適（overfitting）；見 Hutchinson et al. [1994]、Lajbcygier and Connor [1997a,b]、Anders et al. [1998]，以及 Garcia and Gençay [1998, 2000]。
*   相較於股票價格與履約價格，價內程度是一種平穩（stationary）的輸入特徵。使用它有助於模型的泛化能力，並減少過度配適；見 Ghysels et al. [1998] 與 Garcia and Gençay [1998, 2000]。我們自己的實驗也證實，使用價內程度能顯著提升泛化能力。

Bennell 和 Sutcliffe [2004] 針對各種輸入特徵的選擇進行系統性實驗，包括標的資產價格、履約價格、價內外程度（moneyness），以及輸出變數的選擇，包括選擇權價格與選擇權價格除以履約價格。

除了標的資產價格與履約價格之外，波動率（volatility）也是廣泛使用的輸入特徵。這可以透過多種不同方式實現，其中最相關的做法如下：

*   使用歷史波動率估計值作為特徵。
*   使用波動率指數，例如 VIX 作為特徵。

- 使用隱含波動率作為特徵。
- 使用GARCH模型對（已實現或隱含）波動率之預測值作為特徵。

表 2 列出更多波動率相關特徵。不同論文所選擇的特徵整理於表 1 的「Features」欄位中。也有若干論文完全不使用任何波動率類型特徵作為人工神經網路的輸入。

少數論文，例如 Blynski and Faseruk [2006]、Andreou et al. [2008] 或 Wang [2009b]，比較了不同的波動率特徵。此處我們總結其結果。Blynski and Faseruk [2006] 顯示，當使用歷史波動率作為輸入時，人工神經網路優於傳統 Black-Scholes 模型，但使用隱含波動率時則表現較差。Andreou et al. [2008] 指出，將歷史波動率替換為隱含波動率可提升人工神經網路的表現。Wang [2009a] 則主張，使用 GARCH 波動率預測值作為特徵的人工神經網路，優於使用歷史波動率與隱含波動率作為特徵的模型。

部分論文探討額外特徵是否能幫助人工神經網路進行預測。舉例而言，Ghaziri et al. [2000] 與 Healy et al. [2002] 納入了選擇權未平倉量。Samur and Temur [2009] 研究加入變異數是否能改善人工神經網路的表現。Montesdeoca and Niranjan [2016] 探討交易量、選擇權未平倉量及其他變數的潛在預測能力。Cao et al. [2019] 則研究使用標的資產報酬率所能帶來的效益。

## 2.2 輸出變數

表 1 中的論文亦可依其輸出變數進行分類：

- 最常見的輸出是選擇權價格。依是否使用價內外程度（moneyness），或分別使用標的資產價格與履約價格，輸出可能是選擇權價格，或是選擇權價格除以履約價格。部分論文也探討人工神經網路學習所謂「偏差」（bias）的能力，即市場價格與參數模型估計價格之間的差異。此類人工神經網路稱為混合人工神經網路（hybrid ANN）；例如可參見 Boek et al. [1995] 或 Lajbcygier and Connor [1997a,a]。雖然大多數早期論文訓練人工神經網路以擬合價格，但 Garcia and Gençay [2000] 以價格進行訓練，卻以避險誤差進行驗證，以決定能產生最低避險誤差的網路規模。Andreou et al. [2010] 強調，當目標為避險任務時，選擇適當的損失函數至關重要。

- 另一類輸出是隱含波動率（implied volatility）。所得到的隱含波動率可透過 Black-Scholes 公式轉換為選擇權價格。Mostafa and Dillon [2008] 比較了輸出選擇權價格的人工神經網路與輸出隱含波動率的人工神經網路。最近，Liu et al. [2019b] 評估了人工神經網路逼近 Black-Scholes 公式反函數的能力。

- 第三類輸出（在表 1 中皆標示為 HR）是敏感度或避險比率（hedging ratio）。僅有少數論文討論此類人工神經網路架構。最早的論文包括 Carverhill and Cheuk [2003]、Chen and Sutcliffe [2012] 以及 Shin and Ryu [2012]。近年來，Buehler et al. [2019a,b] 與 Ruf and Wang [2020] 延續了這一研究方向。Buehler et al. [2019b] 亦考慮了障礙選擇權（barrier options）等奇異選擇權的避險問題。

我們原本也可將所謂的校準（calibration）論文加入表 1，這些論文建構人工神經網路以將價格對應到特定模型參數，或進行反向對應。但我們決定將這些論文獨立置於下文 4.1 節討論。

## 2.3 績效衡量指標與基準模型

當評估人工神經網路（ANNs）的效能時，常見的統計指標包括平均絕對誤差（mean absolute error, MAE）、平均絕對百分比誤差（mean absolute percentage error, MAPE）以及均方誤差（mean squared error, MSE）<sup>21</sup>。這些指標與評估

<sup>21</sup>有幾篇論文使用表 4 中指標的等價版本。例如，有時會使用均方根誤差（root mean squared error）而非均方誤差。為求一致性，我們在表 1 中已進行相應調整。

17

在單一期間內，針對定價或避險進行評估。有些論文也提出在多個期間評估人工神經網路（ANN）的表現。例如，Hutchinson et al. [1994] 引入了平均絕對追蹤誤差（mean absolute tracking error, MATE）與預測誤差（prediction error, PE），這兩個指標也出現在許多後續論文中。Buehler et al. [2019a] 則引入條件風險價值（conditional value-at-risk, CVaR）來評估避險策略。

人工神經網路的表現也應與基準模型進行比較，例如參數化定價模型。最廣泛使用的基準是 Black-Scholes 公式，該公式需要輸入波動率。如表 1 所總結，歷史波動率估計是最常被採用的方法。文獻中也出現某些隱含波動率（例如歷史隱含波動率或平價隱含波動率）。Blynski and Faseruk [2006] 比較了 Black-Scholes 基準模型中使用歷史實現波動率與歷史隱含波動率的情況。

針對避險任務，以合約特定隱含波動率（contract-specific implied volatility）輸入的 Black-Scholes 公式是一個有效的基準。然而，對於定價任務而言，此一基準將導致誤差為零，因為根據隱含波動率的定義，它對選擇權的定價本來就沒有誤差。因此，對於定價任務，使用合約特定隱含波動率的 Black-Scholes 公式並非適當的基準。

除了 Black-Scholes 公式之外，其他廣泛使用的參數化基準還包括隨機波動率定價模型，例如 Gençay and Gibson [2007]、Jang and Lee [2019] 或 Liu et al. [2019b] 所使用的模型。Ruf and Wang [2020] 觀察到，若所選用的基準同時納入 delta 與 vega 避險，則人工神經網路甚至無法優於一個簡單的雙因子迴歸模型。

針對美式選擇權，所使用的基準包括 Barone-Adesi and Whaley [1987] 定價方法（例如 Lajbcygier [2002]），以及 Cox-Ross-Rubinstein 模型（例如 Chen and Lee [1999]）。

## 2.4 資料分割方法

人工神經網路需要在訓練集（樣本內，in-sample）上進行訓練，然後在測試集（樣本外，out-of-sample）上進行測試。將資料集分割為訓練集與測試集存在多種方法。第一種方法是依時間順序分割，亦即較早期的資料構成訓練集，較晚期的資料構成測試集。表 1 顯示大多數論文採用此一做法。然而，有些研究違反了資料的時間結構，採用不同的分割方式。違反情況包括隨機將資料分割為訓練集與測試集，或是使用所謂的「奇偶分割」（odd-even split）。

隨機分割會破壞時間結構，並在訓練集與測試集之間引入資訊洩漏。當人工神經網路在以此方式建構的訓練集上進行訓練時，測試集上的誤差會低估該網路的泛化誤差（generalisation error）。Yao et al. [2000] 以及我們的 companion paper Ruf and Wang [2020] 對此點有更詳細的討論。

有些論文僅處理來自各種分配的獨立抽樣，因此不涉及任何時間序列結構。雖然這些論文同樣隨機將整個資料集分割為訓練集與測試集，但並未違反時間結構。因此，在表 1 中，我們將此類方法歸類為時間順序分割（chronological partition）。

一個相關的議題是金融資料中時間非齊次性（time-inhomogeneity）的存在；特別是波動率會隨時間改變。在處理真實資料時，有些論文採用滾動視窗（rolling window）方法來解決此問題，尤其當時間範圍較長且未將波動率納入輸入特徵時。此類論文包括 Hutchinson et al. [1994]、Dugas et al. [2009] 等。然而，視窗大小需要多大仍是一個開放性問題。

## 2.5 基礎資產與時間範圍

模擬資料與真實資料皆可用來訓練人工神經網路（ANN）以解決特定問題。模擬資料較易處理，因為它沒有雜訊，且有時可取得接近最佳解的基準，例如 Black-Scholes 與 Heston 模型。例如，le Roux and du Toit [2001]、Morelli et al. [2004] 以及 Karatas et al. [2019] 皆探討了人工神經網路在模擬資料上的表現。大多數其他論文則同時使用模擬資料與真實資料，或僅使用真實資料。S&P500 指數選擇權被最多論文研究，因為它們是交易量最為活絡的選擇權。FTSE100 與 S&P100 指數選擇權也在若干論文中被研究。所有使用到的基礎資產更完整清單請參見表 5。

18

一些論文聚焦於美式選擇權的定價與避險。美式選擇權的標的資產通常為個別股票。涉及美式選擇權的論文包括 Kelly [1994]、Chen and Lee [1999]、Meissner and Kawano [2001]、Pires and Marwala [2004a]、Pires and Marwala [2005] 以及 Amornwattana et al. [2007]。如第 4.3 小節所述，美式選擇權亦可透過人工神經網路（ANN）以不同方式定價，即在動態規劃架構下學習價值函數（value function）或最適停止規則（optimal stopping rule）；參見 Kohler et al. [2010] 與 Becker et al. [2019]。

## 3 推薦論文

在表 1 所列眾多論文中，我們想特別強調其中幾篇。此一選擇顯然帶有個人主觀性。儘管選擇具有主觀性，我們相信此一清單可作為了解此領域的良好起點。我們也提供 Google Scholar 的引用次數。<sup>22</sup> 如前所述，表 1 僅聚焦於使用人工神經網路（ANN）來估計選擇權價格及相關變數的論文。最近在利用人工神經網路進行校準（calibration）或作為計算工具方面已有許多有趣且具前景的發展。這些論文並未納入本文，但第 4 節提供了一些相關文獻的指引。

在以下所強調的論文中，有些是率先提出創新解決方案的論文，其他則以系統性的方式探討該問題。

- Hutchinson et al. [1994]（引用次數：749）是最早使用人工神經網路估計選擇權價格的論文之一，也是被引用最多次的論文。他們提出一種評估多期避險績效的方法論，後續被許多論文採用。
- Lajbcygier and Connor [1997a]（引用次數：<sup>23</sup> 51）是最早提出學習模型價格與觀察到的市場選擇權價格之間差異的論文之一。
- Anders et al. [1998]（引用次數：106）比較了在採用不同波動率估計值時，人工神經網路與 Black-Scholes 基準模型的表現。
- Garcia and Gençay [2000]（引用次數：<sup>24</sup> 210）在人工神經網路中納入了齊次性提示（homogeneity hint）。因此，這是最早將金融領域知識嵌入人工神經網路建構的論文之一。
- Carverhill and Cheuk [2003]（引用次數：15）率先提出一種直接輸出避險策略而非選擇權價格的人工神經網路。
- Bennell and Sutcliffe [2004]（引用次數：83）、Chen and Sutcliffe [2012]（引用次數：12）以及 Hahn [2013]（引用次數：9）提供了三篇廣泛的文獻回顧。
- Dugas et al. [2009]（引用次數：<sup>25</sup> 172）率先設計了一種強制執行無套利條件（no-arbitrage conditions）（例如選擇權價格的凸性）的人工神經網路架構。
- Andreou et al. [2010]（引用次數：19）將人工神經網路與參數模型結合，以學習能回傳隱含模型參數的函數。此類人工神經網路本質上是在校準參數模型。
- Buehler et al. [2019a]（引用次數：23）發展了一個嶄新的架構，用以在存在市場摩擦的情況下為衍生性金融商品投資組合進行避險，並允許以凸風險衡量（convex risk measures）作為損失函數。他們的架構允許在不觀察選擇權價格的情況下進行定價與避險。

由於這是主觀的選擇，我們也想特別提及我們的 companion paper Ruf and Wang [2020]，該論文提供了基於 delta-vega 避險的新基準，並討論了資料外洩（data leakage）議題。

---
<sup>22</sup>截至 2019 年 10 月 3 日。
<sup>23</sup>此計數包含 Lajbcygier and Connor [1997b] 的引用次數。
<sup>24</sup>此計數包含 Garcia and Gençay [1998] 的引用次數。
<sup>25</sup>此計數包含 Dugas et al. [2001] 的引用次數。

19
-----

# 4 **相關論文**

在過去幾年中，已經發展出許多新穎技術，將人工神經網路（ANN）應用於選擇權定價中超越非參數估計價格與避險比率的任務。本節我們針對這個快速發展的文獻提供若干指引。<sup>26</sup>

## 4.1 **校準**

如第 3 節已提及，Andreou et al. [2010] 提出一種人工神經網路，能輸出隱含模型參數。因此，該人工神經網路本質上是在校準參數化模型。我們觀察到近年來將人工神經網路應用於校準的熱潮。在此方法中，選擇權價格首先被對映到一個參數化模型，再由該模型決定選擇權價格。此方法可將計算密集的校準程序移至離線進行，從而大幅加速選擇權定價。

Abu-Mostafa [2001] 使用神經網路校準 Vasicek 模型，並加入一致性提示（consistency hint）以產生有效的參數。最近，Hernandez [2017] 使用人工神經網路校準單因子 Hull-White 模型。Dimitroff et al. [2018]、McGhee [2018] 與 Liu et al. [2019a] 校準隨機波動率模型，而 Stone [2019] 與 Bayer et al. [2019]<sup>27</sup> 則校準粗糙波動率（rough volatility）模型。Itkin [2019] 指出了現有方法中的若干陷阱，並提出解決方案以同時提升校準的效能與準確度。

透過先校準模型、再利用該模型決定避險比率的「間接」方式，至少具有兩項優點。首先，它提供了額外的可解釋性，因為僅以人工神經網路取代校準步驟。這對受監管要求的金融機構而言相當重要。其次，它提供了一種可謂強大的客製化正則化效果，因為它將非參數估計任務替換為估計通常少於 5–10 個參數的模型任務。

## 4.2 **求解偏微分方程式**

選擇權定價問題有時涉及求解偏微分方程式（PDE）。Barucci et al. [1996, 1997] 使用 Galerkin 方法與人工神經網路求解 Black-Scholes 偏微分方程式。E et al. [2017]、Han et al. [2018] 與 Beck et al. [2019] 利用人工神經網路求解高維半線性拋物型偏微分方程式。他們提出使用倒向隨機微分方程式（backward stochastic differential equations）重新表述偏微分方程式，並以人工神經網路逼近未知解的梯度。他們的數值結果顯示，該方法對各種（可能為高維度的）問題均相當有效。其中一個案例研究涉及對 100 個可違約基礎資產的歐式選擇權定價。近期有若干論文進一步發展此一人工神經網路應用，例如 Henry-Labordère [2017]、Sirignano and Spiliopoulos [2018]、Chan-Wai-Nam et al. [2019]、Huré et al. [2019]、Jacquier and Oumgari [2019] 以及 Vidales et al. [2019]。

## 4.3 **逼近最適控制問題中的價值函數**

人工神經網路可用於逼近動態規劃中出現的價值函數，例如出現在美式選擇權定價問題中的價值函數；參見 Ye and Zhang [2019]。Kohler et al. [2010] 使用人工神經網路估計高維度美式選擇權定價的續存價值（continuation values）。Becker et al. [2019] 利用人工神經網路處理最適停止問題，透過蒙地卡羅樣本學習最適停止規則。人工神經網路也被提出用於逼近實質選擇權（real option）定價動態規劃的價值函數，參見 Taudes et al. [1998]。

在這個脈絡下，我們也提及 Fecamp et al. [2019]，他們使用人工神經網路（ANN）作為計算工具，來解決存在交易成本等市場摩擦下的定價與避險問題。

---
<sup>26</sup>有時我們並不容易明確判斷一篇論文是否應歸類於表 1 或本節。例如，第 4.1 節的校準論文，如第 2.2 節所述，本來也可以放入表 1。同樣地，第 4.2 節所討論的 Barucci et al. [1996, 1997] 學習 Black-Scholes 模型，因此原本也可以放入表 1。

<sup>27</sup>更多細節亦可參見 Bayer and Stemper [2018] 以及 Horvath et al. [2019]。

20

# 4.4 **後續研究**

Albanese et al. [2019] 使用人工神經網路（ANN）透過求解分位數迴歸（quantile regression），來計算特定 XVA 計算所需的條件風險值（conditional value-at-risk）與預期虧損（expected shortfall）。

我們也想提及 Halperin [2017] 與 Kolm and Ritter [2019]，他們提出使用強化學習（reinforcement learning）方法，在選擇權定價任務中納入市場摩擦因素。

最後，生成式人工神經網路（generative ANNs）最近被建議作為股價的非參數模擬工具；例如可參見 Henry-Labordère [2019]、Kondratyev and Schwarz [2019]，以及 Wiese et al. [2019b]。此類模擬引擎可進一步用於選擇權定價與避險，這仍是尚待系統性探索的方向。本篇綜述即將完成之際，Wiese et al. [2019a] 提出了一種用於選擇權價格（而非股價）的生成式人工神經網路。

# 5 **題外話：正則化技術**

隨著硬體的進步使得更大的人工神經網路得以建構，正則化技術在人工神經網路訓練中已變得愈發重要。此類技術包括 $L^2$ 正則化、丟棄法（dropout）、提前停止（early stopping）等；參見 Ormoneit [1999]、Gençay and Qi [2001]、Gençay and Salih [2003]，以及 Liu et al. [2019b]。除了這些通用正則化方法外，數篇論文將金融領域知識嵌入人工神經網路中，無論是在架構設計階段或訓練階段。在此也應提及 Lu and Ohta [2003a,b] 所提出的特徵設計，他們在奇異選擇權定價中建議使用數位選擇權價格作為特徵。

在架構設計方面，已被提出的方法包括：

- **齊次性提示（Homogeneity hint）。** Garcia and Gençay [1998, 2000] 透過將人工神經網路分為兩個部分來納入齊次性提示，其中一部分由價內程度（moneyness）控制，另一部分由剩餘到期時間（time-to-maturity）控制。

- **形狀限制輸出（Shape-restricted outputs）。** Dugas et al. [2001, 2009]、Lajbcygier [2004]、Yang et al. [2017]、Huh [2019] 以及 Zheng et al. [2019] 透過固定適當的網路架構，強制人工神經網路定價函數滿足某些無套利條件，例如單調性與凸性。

在訓練階段所使用的技術包括：

- **資料擴增（Data augmentation）。** Yang et al. [2017] 與 Zheng et al. [2019] 產生額外的合成選擇權，以協助人工神經網路的訓練。

- **損失懲罰（Loss penalty）。** Itkin [2019] 與 Ackerer et al. [2019] 在損失函數中加入各種懲罰項。這些懲罰項代表無套利條件。例如，允許日曆套利（calendar arbitrage）的參數配置會受到懲罰。

在人工神經網路訓練的脈絡下，我們也想提及 Niranjan [1996]、de Freitas et al. [2000a,b] 以及 Palmer [2019]。這些論文提出並檢驗了新型的人工神經網路訓練演算法，並在選擇權避險的背景下進行說明；這些演算法包括擴展卡爾曼濾波器（extended Kalman filter）、序列蒙地卡羅方法（sequential Monte Carlo）以及演化演算法（evolutionary algorithms）。

# 參考文獻

Y. S. Abu-Mostafa. Financial model calibration using consistency hints. *IEEE transactions on neural networks*, 12(4):791–808, 2001.

D. Ackerer, N. Tagasovska, and T. Vatter. Deep smoothing of the implied volatility surface. SSRN 3402942, 2019.

21

**References**

P. Ahmed 和 S. Swidler。神經網路生成波動率估計值的預測特性。在 *Decision Technologies for Computational Finance*，頁 247–258，1998。

J. J. Ahn、D. H. Kim、K. J. Oh 和 T. Y. Kim。將選擇權 Greeks 應用於選擇權市場中隱含波動率 (implied volatility) 的方向性預測：一種智慧型方法。《專家系統應用》，39(10)：9315–9322，2012。

C. Albanese、S. Crépey、R. Hoskinson 和 B. Saadeddine。來自資產負債表的 XVA 分析。2019 年 10 月 25 日取自 https://math.maths.univ-evry.fr/crepey/，2019。

H. Amilon。神經網路與 Black–Scholes：定價與避險表現之比較。《預測期刊》，22：317–335，2003。

S. Amornwattana、D. Enke 和 C. H. Dagli。使用神經網路估計波動率的混合選擇權定價模型。《國際一般系統期刊》，36(5)：558–573，2007。

U. Anders、O. Korn 和 C. Schmitt。改善選擇權定價：一種神經網路方法。《預測期刊》，17(5-6)：369–388，1998。

P. C. Andreou。《選擇權定價的參數與非參數函數估計及其在避險與交易之應用》。博士論文，賽普勒斯大學，2008。

P. C. Andreou、C. Charalambous 和 S. H. Martzoukos。選擇權定價方法使用人工神經網路之批判性評估。在 *International Conference on Artificial Neural Networks*，頁 1131–1136，2002。

P. C. Andreou、C. Charalambous 和 S. H. Martzoukos。用於歐式選擇權定價的穩健人工神經網路。《計算經濟學》，27(2-3)：329–351，2006。

P. C. Andreou、C. Charalambous 和 S. H. Martzoukos。結合人工神經網路與帶有隱含參數的參數模型來定價與交易歐式選擇權。《歐洲運營研究期刊》，185(3)：1415–1433，2008。

P. C. Andreou、C. Charalambous 和 S. H. Martzoukos。選擇權定價的廣義參數函數。《銀行與金融期刊》，34(3)：633–646，2010。

M. Avellaneda、A. Carelli 和 F. Stella。遵循貝氏路徑進行選擇權定價。《金融計算智慧期刊》，1998。

G. Barone-Adesi 和 R. E. Whaley。美式選擇權價值的高效解析近似。《金融期刊》，42(2)：301–320，1987。

E. Barucci、U. Cherubini 和 L. Landi。在隨機波動率下使用神經網路進行無套利資產定價。在 *Neural Networks in Financial Engineering: Proceedings of the Third International Conference on Neural Networks in the Capital Markets*，頁 3–16，1996。

E. Barucci、U. Cherubini 和 L. Landi。透過 Galerkin 方法使用神經網路進行或有請求權定價。在 *Computational Approaches to Economic Problems*，頁 127–141，1997。

M. Barunikova 和 J. Barunik。神經網路作為半參數選擇權定價工具。《捷克計量經濟學會通報》，18，2011。

D. S. Bates。跳躍與隨機波動率：德國馬克選擇權隱含的匯率過程。《金融研究評論》，9(1)：69–107，1996。

C. Bayer 和 B. Stemper。粗糙隨機波動率模型的深度校準。arXiv:1810.03399，2018。

C. Bayer、B. Horvath、A. Muguruza、B. Stemper 和 M. Tomas。關於（粗糙）隨機波動率模型的深度校準。arXiv:1908.08806，2019。

C. Beck、S. Becker、P. Cheridito、A. Jentzen 和 A. Neufeld。拋物型偏微分方程的深度分割方法。arXiv:1907.03452，2019。

22

**輸出：**

S. Becker, P. Cheridito, and A. Jentzen. *Deep optimal stopping.* *Journal of Machine Learning Research*, 20 (74):1–25, 2019.

J. Bennell and C. Sutcliffe. Black–Scholes versus artificial neural networks in pricing FTSE 100 options. *Intelligent Systems in Accounting, Finance & Management: International Journal*, 12(4):243–260, 2004.

M. Billio, M. Corazza, and M. Gobbo. Option pricing via regime switching models and multilayer perceptrons: a comparative approach. *Rendiconti per gli Studi Economici Quantitativi*, 2002:39–59, 2002.

L. Blynski and A. Faseruk. Comparison of the effectiveness of option price forecasting: Black–Scholes vs. simple and hybrid neural networks. *Journal of Financial Management & Analysis*, 19(2):46–58, 2006.

C. Boek, P. Lajbcygier, M. Palaniswami, and A. Flitman. A hybrid neural network approach to the pricing of options. In *Proceedings of ICNN'95–International Conference on Neural Networks*, volume 2, pages 813–817. IEEE, 1995.

T. Briegel and V. Tresp. Dynamic neural regression models. Retrieved on August 29, 2019 from <font color=blue>https://epub.ub.uni-muenchen.de/1571/</font>, 2000.

H. Buehler, L. Gonon, J. Teichmann, and B. Wood. Deep hedging. *Quantitative Finance*, 19(8):1271–1291, 2019a.

H. Buehler, L. Gonon, J. Teichmann, B. Wood, B. Mohan, and J. Kochems. Deep hedging: hedging derivatives under generic market frictions using reinforcement learning. SSRN 3355706, 2019b.

M. Can and Š. Fadda. A nonparametric approach to pricing options learning networks. *Southeast Europe Journal of Soft Computing*, 3(1), 2014.

J. Cao, J. Chen, and J. C. Hull. A neural network approach to understanding implied volatility movements. SSRN 3288067, 2019.

A. Carelli, S. Silani, and F. Stella. Profiling neural networks for option pricing. *International Journal of Theoretical and Applied Finance*, 3(02):183–204, 2000.

P. Carr, H. Geman, D. B. Madan, and M. Yor. Stochastic volatility for Lévy processes. *Mathematical Finance*, 13(3):345–382, 2003.

A. P. Carverhill and T. H. Cheuk. Alternative neural network approach for option pricing and hedging. SSRN 480562, 2003.

Q. Chan-Wai-Nam, J. Mikael, and X. Warin. Machine learning for semi linear PDEs. *Journal of Scientific Computing*, 79(3):1667–1712, 2019.

T.-Y. Chang, Y.-H. Wang, and H.-Y. Yeh. Forecasting of option prices using a neural network model. *Journal of Accounting, Finance & Management Strategy*, 8(1):123–136, 2013.

C. Charalambous and S. H. Martzoukos. Hybrid artificial neural networks for efficient valuation of real options and financial derivatives. *Computational Management Science*, 2(2):155–161, 2005.

F. Chen and C. Sutcliffe. Pricing and hedging short sterling options using neural networks. *Intelligent Systems in Accounting, Finance and Management*, 19(2):128–149, 2012.

J. Chen. Learning the Black–Scholes formula via support vector machines. In *Recent Advances in Statistics Application and Related Areas, 2nd Conference of the International Institute of Applied Statistics Studies*, volume 1&2, pages 756–760, 2009.

S.-H. Chen and W.-C. Lee. Pricing call warrants with artificial neural networks: the case of the Taiwan derivative market. In *IJCNN'99. International Joint Conference on Neural Networks. Proceedings (Cat. No. 99CH36339)*, volume 6, pages 3877–3882. IEEE, 1999.

D.-Y. Chiu and C.-C. Lin. Exploring internal mechanism of warrant in financial market with a hybrid approach. *Expert Systems with Applications*, 35(3):1237–1245, 2008.

23

**Output:**

H.-J. Choi, H.-S. Lee, G.-S. Han, 和 J. Lee. *Efficient option pricing via a globally regularized neural network*. In *International Symposium on Neural Networks*, pages 988–993, 2004.

C. J. Corrado 和 T. Su. *Skewness and kurtosis in S&P 500 index returns implied by option prices*. *Journal of Financial Research*, 19(2):175–192, 1996.

J. C. Cox, S. A. Ross, 和 M. Rubinstein. *Option pricing: a simplified approach*. *Journal of Financial Economics*, 7(3):229–263, 1979.

R. Culkin 和 S. R. Das. *Machine learning in finance: the case of deep learning for option pricing*. *Journal of Investment Management*, 15(4):92–100, 2017.

S. P. Das 和 S. Padhy. *A new hybrid parametric and machine learning model with homogeneity hint for European-style index option pricing*. *Neural Computing and Applications*, 28(12):4061–4077, 2017.

J. F. G. de Freitas, M. Niranjan, 和 A. H. Gee. *Hierarchical Bayesian models for regularization in sequential learning*. *Neural Computation*, 12(4):933–953, 2000a.

J. F. G. de Freitas, M. Niranjan, A. H. Gee, 和 A. Doucet. *Sequential Monte Carlo methods to train neural network models*. *Neural Computation*, 12(4):955–993, 2000b.

G. Dimitroff, D. Röder, 和 C. Fries. *Volatility model calibration with convolutional neural networks*. SSRN 3252432, 2018.

Z. A. Dindar 和 T. Marwala. *Option pricing using a committee of neural networks and optimized networks*. In *2004 IEEE International Conference on Systems, Man and Cybernetics (IEEE Cat. No. 04CH37583)*, volume 1, pages 434–438. IEEE, 2004.

C. Dugas, Y. Bengio, F. Bélisle, C. Nadeau, 和 R. Garcia. *Incorporating second-order functional knowledge for better option pricing*. In *Advances in Neural Information Processing Systems*, pages 472–478, 2001.

C. Dugas, Y. Bengio, F. Bélisle, C. Nadeau, 和 R. Garcia. *Incorporating functional knowledge in neural networks*. *Journal of Machine Learning Research*, 10(Jun):1239–1262, 2009.

W. E, J. Han, 和 A. Jentzen. *Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations*. *Communications in Mathematics and Statistics*, 5(4):349–380, 2017.

Z. Fang 和 K. George. *Application of machine learning: an analysis of Asian options pricing using neural network*. In *2017 IEEE 14th International Conference on e-Business Engineering (ICEBE)*, pages 142–149. IEEE, 2017.

S. Fecamp, J. Mikael, 和 X. Warin. *Risk management with machine-learning-based algorithms*. arXiv:1902.05287, 2019.

R. Ferguson 和 A. Green. *Deeply learning derivatives*. SSRN 3244821, 2018.

J. Galindo-Flores. *A framework for comparative analysis of statistical and machine learning methods: an application to the Black–Scholes option pricing model*. *Computational Finance 1999*, pages 635–660, 2000.

R. Garcia 和 R. Gençay. *Option pricing with neural networks and a homogeneity hint*. In *Decision Technologies for Computational Finance*, pages 195–205, 1998.

R. Garcia 和 R. Gençay. *Pricing and hedging derivative securities with neural networks and a homogeneity hint*. *Journal of Econometrics*, 94(1-2):93–115, 2000.

J. Gatheral 和 A. Jacquier. *Arbitrage-free SVI volatility surfaces*. *Quantitative Finance*, 14(1):59–71, 2014.

D. S. Geigle. *An Artificial Neural Network Approach to the Valuation of Options and Forecasting of Volatility*. 博士論文，Nova Southeastern University, 1999。

24

**References**

D. S. Geigle 與 J. E. Aronson. *An artificial neural network approach to the valuation of options and forecasting of volatility.* *Journal of Computational Intelligence in Finance*, 7(6):19–25, 1999.

R. Gençay 與 R. Gibson. *Model risk for European-style stock index options.* *IEEE Transactions on Neural Networks*, 18(1):193–202, 2007.

R. Gençay 與 M. Qi. *Pricing and hedging derivative securities with neural networks: Bayesian regularization, early stopping, and bagging.* *IEEE Transactions on Neural Networks*, 12(4):726–734, 2001.

R. Gençay 與 A. Salih. *Degree of mispricing with the Black–Scholes model and nonparametric cures.* *Annals of Economics and Finance*, 4:73–101, 2003.

H. Ghaziri、S. Elfakhani 與 J. Assi. *Neural networks approach to pricing options.* *Neural Network World*, 10(1):271–277, 2000.

J. Ghosn 與 Y. Bengio. *Multi-task learning for option pricing.* 2019 年 10 月 29 日取自 https://cirano.qc.ca/files/publications/2002s-53.pdf, 2002.

E. Ghysels、V. Patilea、É. Renault 與 O. Torrès. *Nonparametric methods and option pricing.* 收錄於 D. Hand 與 S. Jacka 編，《Statistics in Finance》，第 13 章，頁 261–282。John Wiley & Sons, 1998.

N. Gradojevic 與 D. Kukolj. *Parametric option pricing: a divide-and-conquer approach.* *Physica D: Nonlinear Phenomena*, 240(19):1528–1535, 2011.

N. Gradojevic、R. Gençay 與 D. Kukolj. *Option pricing with modular neural networks.* *IEEE Transactions on Neural Networks*, 20(4):626–637, 2009.

A. Gregoriou、J. Healy 與 C. Ioannidis. *Hedging under the influence of transaction costs: an empirical investigation on FTSE 100 index options.* *Journal of Futures Markets*, 27(5):471–494, 2007.

J. T. Hahn. *Option Pricing Using Artificial Neural Networks: An Australian Perspective.* 博士論文，Bond University, 2013.

I. Halperin. *QLBS: Q-learner in the Black-Scholes (-Merton) worlds.* arXiv:1712.04609, 2017.

S. A. Hamid 與 A. Habib. *Can neural networks learn the Black-Scholes model?: A simplified approach.* 2019 年 9 月 9 日取自 https://academicarchive.snhu.edu/bitstream/handle/10474/1662/cfs2005-01.pdf, 2005.

J. Han、A. Jentzen 與 W. E. *Solving high-dimensional partial differential equations using deep learning.* *Proceedings of the National Academy of Sciences*, 115(34):8505–8510, 2018.

M. Hanke. *Neural network approximation of option pricing formulas for analytically intractable option pricing models.* *Journal of Computational Intelligence in Finance*, 5(5):20–27, 1997.

M. Hanke. *Adaptive hybrid neural network option pricing.* *Journal of Computational Intelligence in Finance*, 7(5):33–39, 1999a.

M. Hanke. *Neural networks versus Black-Scholes: an empirical comparison of the pricing accuracy of two fundamentally different option pricing methods.* *Journal of Computational Intelligence in Finance*, 5: 26–34, 1999b.

J. Healy、M. Dixon、B. Read 與 F. Cai. *A data-centric approach to understanding the pricing of financial options.* *The European Physical Journal B*, 27(2):219–227, 2002.

J. V. Healy、M. Dixon、B. J. Read 與 F. F. Cai. *Confidence in data mining model predictions: a financial engineering application.* 收錄於 *IECON'03. 29th Annual Conference of the IEEE Industrial Electronics Society (IEEE Cat. No. 03CH37468)*，第 2 卷，頁 1926–1931。IEEE, 2003.

J. V. Healy、M. Dixon、B. J. Read 與 F. F. Cai。《選擇權價格資料探勘模型的信心限界》（*Confidence limits for data mining models of options prices.*）《物理學報 A：統計力學及其應用》（*Physica A: Statistical Mechanics and its Applications*），344(1-2):162–167，2004 年。

J. V. Healy、M. Dixon、B. J. Read 與 F. F. Cai。《隱含資產價格分布的非參數萃取》（*Non-parametric extraction of implied asset price distributions.*）《物理學報 A：統計力學及其應用》（*Physica A: Statistical Mechanics and its Applications*），382(1):121–128，2007 年。

25

**輸出：**

P. Henry-Labordère. 深度原對偶演算法於 BSDEs：機器學習於 CVA 與 IM 之應用。SSRN 3071506，2017。

P. Henry-Labordère. 金融資料的生成模型。SSRN 3408007，2019。

A. Hernandez. 以類神經網路進行模型校準。*Risk Magazine*，頁 1–5，2017 年 6 月。

R. Herrmann 和 A. Narr. 類神經網路與衍生性金融商品評價：德國股票指數選擇權隱含定價機制的一些洞見。2019 年 8 月 29 日取自 http://finance.fbv.kit.edu/download/dp202.pdf，1997。

S. L. Heston. 隨機波動率選擇權的封閉解及其於債券與貨幣選擇權之應用。*The Review of Financial Studies*，6(2):327–343，1993。

B. Horvath, A. Muguruza, 和 M. Tomas. 深度學習波動率 (deep learning volatility)。arXiv:1901.09647，2019。

S.-C. Huang. 使用無跡卡爾曼濾波器與支援向量機進行選擇權價格線上預測。*Expert Systems with Applications*，34(4):2819–2825，2008。

S.-C. Huang 和 T.-K. Wu. 混合無跡卡爾曼濾波器與支援向量機模型於選擇權價格預測之應用。在 *International Conference on Natural Computation*，頁 303–312，2006。

J. Huh. 以指數 Lévy 類神經網路定價選擇權。*Expert Systems with Applications*，127:128–140，2019。

J. Hull 和 A. White. 選擇權的最適 delta 避險。*Journal of Banking & Finance*，82:180–190，2017。

C. Huré, H. Pham, 和 X. Warin. 高維非線性偏微分方程的一些機器學習架構。arXiv:1902.01599，2019。

J. M. Hutchinson, A. W. Lo, 和 T. Poggio. 以學習網路對衍生性金融商品進行非參數定價與避險。*The Journal of Finance*，49(3):851–889，1994。

A. Itkin. 選擇權定價模型的深度學習校準：若干陷阱與解決方案。arXiv:1906.03507，2019。

A. Jacquier 和 M. Oumgari. 粗糙局部隨機波動率的深度 PPDE。arXiv:1906.02551，2019。

H. Jang 和 J. Lee. 用於風險中性定價美國指數選擇權的生成貝氏類神經網路模型。*Quantitative Finance*，19(4):587–603，2019。

K.-H. Jung, H.-C. Kim, 和 J. Lee. 具信心區間資訊的選擇權定價新型學習網路。在 *International Symposium on Neural Networks*，頁 491–497，2006。

M. Kakati. 人工類神經網路在印度股票選擇權市場的定價與避險表現。*The ICFAI Journal of Applied Finance*，11(1):62–73，2005。

M. Kakati. 使用自適應類神經模糊系統 (ANFIS) 進行選擇權定價。*ICFAI Journal of Derivatives Markets*，5(2)，2008。

O. Karaali, W. Edelberg, 和 J. Higgins. 使用類神經網路建模波動率衍生性商品。在 *Proceedings of the IEEE/IAFE 1997 Computational Intelligence for Financial Engineering*，頁 280–286。IEEE，1997。

T. Karatas, A. Oskoui, 和 A. Hirsa. 在各種不同過程下用於香草／奇異選擇權定價／校準的監督式深度類神經網路 (DNNs)。arXiv:1902.05810，2019。

D. L. Kelly. 使用類神經網路評價與避險美國賣權。2019 年 8 月 29 日取自 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.8497&rep=rep1&type=pdf，1994。

B.-H. Kim, D. Lee, 和 J. Lee. 使用重建徑向基函數網路逼近局部波動率函數。在 *International Symposium on Neural Networks*，頁 524–530，2006。

26

P. Ko, P. Lin, W. Chien, and Y. Cheng. Hedging derivative securities based on the neural network coefficient model. In *Proceedings of the Eighth Joint Conference on Information Sciences*, pages 1163–1166, 2005.

P.-C. Ko. 選擇權評價基於類神經迴歸模型 (Option valuation based on the neural regression model). *Expert Systems with Applications*, 36(1): 464–471, 2009.

M. Kohler, A. Krzyżak, and N. Todorovic. 以類神經網路訂價高維度美式選擇權 (Pricing of high-dimensional American options by neural networks). *Mathematical Finance*, 20(3):383–410, 2010.

P. N. Kolm and G. Ritter. 動態複製與避險：強化學習方法 (Dynamic replication and hedging: A reinforcement learning approach). *The Journal of Financial Data Science*, 1(1):159–171, 2019.

A. Kondratyev and C. Schwarz. 市場生成器 (The market generator). SSRN 3384948, 2019.

S. G. Kou. 選擇權訂價的跳躍擴散模型 (A jump-diffusion model for option pricing). *Management Science*, 48(8):1086–1101, 2002.

J. Krause. 以類神經網路進行選擇權訂價 (Option pricing with neural networks). In *Proceedings of the Fourth European Congress on Intelligent Techniques and Soft Computing*, volume 3, pages 2206–2210, 1996.

G. Lachtermacher and L. Rodrigues Gaspar. 類神經網路在巴西資本市場衍生性證券訂價預測之應用 (Neural networks in derivative securities pricing forecasting in Brazilian capital markets). In *Neural Networks in Financial Engineering: Proceedings of the Third International Conference on Neural Networks in the Capital Markets*, pages 92–97, 1996.

W.-N. Lai. 估計選擇權隱含風險中性密度的方法比較 (Comparison of methods to estimate option implied risk-neutral densities). *Quantitative Finance*, 14(10):1839–1855, 2014.

P. R. Lajbcygier. 比較傳統模型與類神經網路模型於選擇權訂價 (Comparing conventional and artificial neural network models for the pricing of options). In *Neural Networks in Business: Techniques and Applications*, pages 220–235. IGI Global, 2002.

P. R. Lajbcygier. 以乘積限制混合類神經網路改善選擇權訂價 (Improving option pricing with the product constrained hybrid neural network). In *Artificial Neural Networks and Neural Information Processing*, pages 615–621, 2003.

P. R. Lajbcygier. 以乘積限制混合類神經網路改善選擇權訂價 (Improving option pricing with the product constrained hybrid neural network). *IEEE Transactions on Neural Networks*, 15(2):465–476, 2004.

P. R. Lajbcygier and J. T. Connor. 以類神經網路與拔靴法改善選擇權訂價 (Improved option pricing using artificial neural networks and bootstrap methods). *International Journal of Neural Systems*, 8(04):457–471, 1997a.

P. R. Lajbcygier and J. T. Connor. 以拔靴法改善選擇權訂價 (Improved option pricing using bootstrap methods). In *Proceedings of International Conference on Neural Networks*, volume 4, pages 2193–2197. IEEE, 1997b.

P. R. Lajbcygier and A. Flitman. 使用最適隱含波動率之非參數迴歸技術在選擇權訂價上的比較 (A comparison of non-parametric regression techniques for the pricing of options using an optimal implied volatility). In *Decision Technologies for Financial Engineering: Proceedings of the Fourth International Conference on Neural Networks in Capital Markets*, pages 201–213, 1996.

P. R. Lajbcygier, C. Boek, A. Flitman, and M. Palaniswami. 比較傳統模型與類神經網路模型於期貨選擇權訂價 (Comparing conventional and artificial neural network models for the pricing of options on futures). *NeuroVe$t Journal*, 4(5):16–24, 1996a.

P. R. Lajbcygier, C. Boek, M. Palaniswami, and A. Flitman. 類神經網路對所有普通股指數期貨選擇權 (SPI options on futures) 之訂價 (Neural network pricing of all ordinaries SPI options on futures). In *Neural Networks in Financial Engineering: Proceedings of the Third International Conference on Neural Networks in the Capital Markets*, 1996b.

P. R. Lajbcygier, A. Flitman, A. Swan, and R. J. Hyndman. 使用混合類神經網路模型結合歷史波動率進行選擇權訂價與交易 (The pricing and trading of options using a hybrid neural network model with historical volatility). *NeuroVe$t Journal*, pages 27–41, 1997.

L. J. le Roux and G. S. du Toit. 以類神經網路模擬 Black & Scholes 模型 (Emulating the Black & Scholes model with a neural network). *Southern African Business Review*, 5(1):54–57, 2001.

M. T. Leung, A.-S. Chen, and R. Mancha. 利用資訊內容建構新型神經網路集成以進行金融工程衍生性商品交易決策。*Intelligent Systems in Accounting, Finance & Management*, 16(4):257–277, 2009。

27

X. Liang, H. Zhang, and J. Yang. Pricing options in Hong Kong market based on neural networks. *In International Conference on Neural Information Processing*, pages 410–419, 2006.

X. Liang, H. Zhang, J. Xiao, and Y. Chen. Improving option price forecasts with neural networks and support vector regressions. *Neurocomputing*, 72(13-15):3055–3065, 2009.

C.-T. Lin and H.-Y. Yeh. 臺灣股票指數選擇權價格之評價——Black-Scholes 模型與類神經網路模型績效之比較 (The valuation of Taiwan stock index option price—comparison of performances between Black-Scholes and neural network model). *Journal of Statistics and Management Systems*, 8(2): 355–367, 2005.

D. Liu and S. Huang. The performance of hybrid artificial neural network models for option pricing during financial crises. *Journal of Data Science*, 14(1):1–18, 2016.

D. Liu and L. Zhang. Pricing Chinese warrants using artificial neural networks coupled with Markov regime switching model. *International Journal of Financial Markets and Derivatives*, 2(4):314–330, 2011.

M. Liu. Option pricing with neural networks. In *Progress in Neural Information Processing*, volume 2, pages 760–765, 1996.

S. Liu, A. Borovykh, L. A. Grzelak, and C. W. Oosterlee. A neural network-based framework for financial model calibration. *Journal of Mathematics in Industry*, Forthcoming, 2019a.

S. Liu, C. W. Oosterlee, and S. M. Bohte. Pricing options and computing implied volatilities using neural networks. *Risks*, 7(1):1–22, 2019b.

X. Liu, Y. Cao, C. Ma, and L. Shen. Wavelet-based option pricing: an empirical study. *European Journal of Operational Research*, 272(3):1132–1142, 2019c.

F. A. Longstaff and E. S. Schwartz. Valuing American options by simulation: a simple least-squares approach. *The Review of Financial Studies*, 14(1):113–147, 2001.

J. Lu and H. Ohta. A data and digital-contracts driven method for pricing complex derivatives. *Quantitative Finance*, 3(3):212–219, 2003a.

J. Lu and H. Ohta. Digital contracts-driven method for pricing complex derivatives. *Journal of the Operational Research Society*, 54(9):1002–1010, 2003b.

M. Ludwig. Robust estimation of shape-constrained state price density surfaces. *The Journal of Derivatives*, 22(3):56–72, 2015.

D. B. Madan, P. P. Carr, and E. C. Chang. The variance Gamma process and option pricing. *Review of Finance*, 2(1):79–105, 1998.

M. Malliaris and L. Salchenberger. Beating the best: a neural network challenges the Black-Scholes formula. In *Proceedings of 9th IEEE Conference on Artificial Intelligence for Applications*, pages 445–449. IEEE, 1993a.

M. Malliaris and L. Salchenberger. A neural network model for estimating option prices. *Journal of Applied Intelligence*, 3(3):193–206, 1993b.

M. Malliaris and L. Salchenberger. Using neural networks to forecast the S&P100 implied volatility. *Neurocomputing*, 10(2):183–195, 1996.

C. G. Martel, M. D. G. Artiles, and F. F. Rodriguez. A financial option pricing model based on learning algorithms. In *Proceedings of the World Multiconference on Applied Economics, Business and Development*, pages 153–157, 2009.

W. A. McGhee. An artificial neural network representation of the SABR stochastic volatility model. SSRN 3288882, 2018.

G. Meissner and N. Kawano. Capturing the volatility smile of options on high-tech stocks—a combined GARCH-neural network approach. *Journal of Economics and Finance*, 25(3):276–292, 2001.

28

**F. G. Miranda 與 N. Burgess。** *使用類神經網路方法進行選擇權定價的日內波動率預測*。收錄於 *Proceedings of 1995 Conference on Computational Intelligence for Financial Engineering*，第 31 頁。IEEE，1995。

S. K. Mitra。提升選擇權價格估計準確度的人工類神經網路方法。SSRN 876881，2006。

S. K. Mitra。一種結合類神經網路方法與 Black-Scholes 公式的選擇權定價模型。*Global Journal of Computer Science and Technology*，12(4)，2012。

G. Montagna、M. Morelli、O. Nicrosini、P. Amato 與 M. Farina。利用路徑積分與類神經網路進行衍生性金融商品定價。*Physica A: Statistical Mechanics and its Applications*，324(1-2):189–195，2003。

L. Montesdeoca 與 M. Niranjan。擴展資料驅動人工類神經網路金融選擇權定價模型的特徵集合。收錄於 *2016 IEEE Symposium Series on Computational Intelligence (SSCI)*，第 1–6 頁。IEEE，2016。

M. J. Morelli、G. Montagna、O. Nicrosini、M. Treccani、M. Farina 與 P. Amato。使用類神經網路進行金融衍生性商品定價。*Physica A: Statistical Mechanics and its Applications*，338(1-2):160–165，2004。

F. Mostafa。*類神經網路在市場風險上的應用*。博士論文，Curtin University，2011。

F. Mostafa 與 T. Dillon。類神經網路於選擇權定價之應用。*WIT Transactions on Information and Communication Technologies*，41:71–85，2008。

M. Niranjan。運用模型基礎與類神經網路方法進行金融選擇權定價的序列追蹤。收錄於 *Advances in Neural Information Processing Systems*，第 960–966 頁，1996。

D. Ormoneit。連續學習的正則化方法及其於金融衍生性商品定價之應用。*Neural Networks*，12(10):1405–1412，1999。

S. Palmer。*演化演算法與衍生性金融商品定價的計算方法*。博士論文，University College London，2019。

S. Palmer 與 D. Gorse。使用蒙地卡羅模擬與繁殖 PSO 訓練類神經網路的擬解析隨機選擇權定價解。收錄於 *European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning*，第 365–370 頁，2017。

A. Pande 與 R. Sahu。股利支付股票波動率估計與選擇權價格預測的新方法。收錄於 *WEHIA 2006–1st International Conference on Economic Sciences with Heterogeneous Interacting Agents; 15–17 June 2006, University of Bologna, Italy*，2006。

H. Park、N. Kim 與 J. Lee。參數模型與非參數機器學習模型在選擇權價格預測上的實證比較研究：以 KOSPI 200 指數選擇權為例。*Expert Systems with Applications*，41(11):5227–5237，2014。

B. Phani、B. Chandra 與 V. Raghav。追求高效選擇權定價預測模型的機器學習技術探索。收錄於 *The 2011 International Joint Conference on Neural Networks*，第 654–657 頁。IEEE，2011。

M. M. Pires 與 T. Marwala。使用多層感知器與支援向量機進行美式選擇權定價。收錄於 *2004 IEEE International Conference on Systems, Man and Cybernetics (IEEE Cat. No. 04CH37583)*，第 2 卷，第 1279–1285 頁。IEEE，2004a。

M. M. Pires 與 T. Marwala。使用貝氏類神經網路進行選擇權定價。收錄於 *Fifteenth Annual Symposium of the Pattern Recognition Association of South Africa*，第 161–166 頁，2004b。

M. M. Pires 與 T. Marwala。使用貝氏多層感知器與貝氏支持向量機進行美式選擇權定價。在 *IEEE 3rd International Conference on Computational Cybernetics*，頁 219–224。IEEE，2005 年。

29

M. Qi. *Financial Applications of Generalized Nonlinear Nonparametric Econometric Methods* (*Artificial Neural Networks*). 博士論文，Ohio State University，1996 年。

M. Qi 和 G. Maddala。使用人工神經網路進行選擇權定價：S&P 500 指數買權的案例。在 *Neural Networks in Financial Engineering: Proceedings of the Third International Conference on Neural Networks in the Capital Markets*，頁 78–91，1996 年。

C. Quek、M. Pasquier 和 N. Kumar。一種基於遞迴神經網路的新穎預測系統，用於選擇權交易與避險。*Applied Intelligence*，29(2):138–151，2008 年。

M. Raberto、G. Cuniberti、M. Riani、E. Scales、F. Mainardi 和 G. Servizi。在稀有事件存在下學習短期選擇權評價。*International Journal of Theoretical and Applied Finance*，3(03):563–564，2000 年。

J. Ruf 和 W. Wang。使用神經網路避險。SSRN 3580132，2020 年。

S. Saito 和 L. Jun。與 Black 和 Scholes 模型相關的神經網路選擇權定價。在 *Proceedings of the Fifth Conference of the Asian Pacific Operations Research Society*，2000 年。

Z. I. Samur 和 G. T. Temur。人工神經網路在選擇權定價的應用：S&P 100 指數選擇權的案例。*International Journal of Social, Behavioral, Educational, Economic, Business and Industrial Engineering*，3(6):644–649，2009 年。

A. Saxena。S&P CNX Nifty 選擇權評價：Black-Scholes 與混合 ANN 模型之比較。在 *Proceedings SAS Global Forum*，2008 年。

C. Schittenkopf 和 G. Dorffner。從選擇權價格萃取風險中性密度：使用混合密度網路改善定價。*IEEE Transactions on Neural Networks*，12(4):716–725，2001 年。

H. J. Shin 和 J. Ryu。使用人工神經網路進行選擇權交易的動態避險策略。*International Journal of Software Engineering and its Applications*，6(4):111–116，2012 年。

J. Sirignano 和 K. Spiliopoulos。DGM：一種求解偏微分方程的深度學習演算法。*Journal of Computational Physics*，375:1339–1364，2018 年。

H. Stone。校準粗糙波動率模型：一種卷積神經網路方法。*Quantitative Finance*，頁 1–14，2019 年。

A. Taudes、M. Natter 和 M. Trcka。使用神經網路的實質選擇權評價。*Intelligent Systems in Accounting, Finance & Management*，7(1):43–52，1998 年。

S. D. Teddy、E.-K. Lai 和 C. Quek。一種受大腦啟發的小腦關聯記憶方法，用於選擇權定價與套利交易。在 *International Conference on Neural Information Processing*，頁 370–379，2006 年。

S. D. Teddy、E.-K. Lai 和 C. Quek。一種小腦關聯記憶方法，用於選擇權定價與套利交易。*Neurocomputing*，71(16-18):3303–3315，2008 年。

N. S. Thomaidis、V. S. Tzastoudis 和 G. Dounias。神經網路模型選擇策略在 S&P500 股票指數選擇權定價上的比較。*International Journal on Artificial Intelligence Tools*，16(06):1093–1113，2007 年。

R. Tsaih。《敏感度分析、神經網路與財務》。在 *IJCNN'99. International Joint Conference on Neural Networks. Proceedings (Cat. No. 99CH36339)*，第 6 卷，頁 3830–3835。IEEE，1999 年。

C.-H. Tseng、S.-T. Cheng、Y.-H. Wang 和 J.-T. Peng。臺灣股票指數選擇權價格之混合 EGARCH 波動率的人工神經網路模型。*Physica A: Statistical Mechanics and its Applications*，387(13):3192–3200，2008 年。

W. L. Tung 和 C. Quek。GenSo-OPATS：一個受大腦啟發的動態演化選擇權定價模型與套利交易系統。在 *2005 IEEE Congress on Evolutionary Computation*，第 3 卷，頁 2429–2436。IEEE，2005 年。

30

**References**

W. L. Tung 與 C. Quek。使用自組織神經模糊語義網路與跨式選擇權方法進行金融波動率交易。*Expert Systems with Applications*, 38(5):4668–4688, 2011。

V. S. Tzastoudis, N. S. Thomaidis, 與 G. D. Dounias。改善基於神經網路的選擇權價格預測。在 *Hellenic Conference on Artificial Intelligence*，頁 378–388，2006。

M. S. Vidales, D. Siska, 與 L. Szpruch。參數偏微分方程的無偏差深度求解器。arXiv:1810.05094, 2019。

C. von Spreckelsen, H.-J. von Mettenheim, 與 M. H. Breitner。朝向高頻金融決策支援系統以神經網路為貨幣期貨選擇權定價的步驟。*International Journal of Applied Decision Sciences*, 7(3):223–238, 2014。

C.-P. Wang, S.-H. Lin, H.-H. Huang, 與 P.-C. Wu。使用神經網路在不同波動率模型下預測臺灣選擇權（TXO）價格。*Expert Systems with Applications*, 39(5):5025–5032, 2012。

H.-W. Wang。使用演化資料探勘進行雙衍生性商品價差交易與避險。*Journal of American Academy of Business*, 9:45–52, 2006。

P. Wang。以支援向量迴歸及帶跳躍的隨機波動率模型為貨幣選擇權定價。*Expert Systems with Applications*, 38(1):1–7, 2011。

Y.-H. Wang。股票指數選擇權價格的非線性神經網路預測模型：混合 GJR–GARCH 方法。*Expert Systems with Applications*, 36(1):564–570, 2009a。

Y.-H. Wang。使用神經網路預測股票指數選擇權價格：一種新的混合 GARCH 方法。*Quality & Quantity*, 43(5):833–843, 2009b。

*A. White.* *以期貨式保證金制度為選擇權定價：遺傳適應性神經網路方法*。Garland Publishing, 2000。

A. J. White。一種遺傳適應性神經網路方法用於選擇權定價：模擬分析。*Journal of Computational Intelligence in Finance*, 6(2):13–23, 1998。

M. Wiese, L. Bai, B. Wood, 與 H. Buehler。深度避險：學習模擬股票選擇權市場。arXiv:1911.01700, 2019a。

M. Wiese, R. Knobloch, R. Korn, 與 P. Kretschmer。Quant GANs：金融時間序列的深度生成。arXiv:1907.06673, 2019b。

L. Xu, M. Dixon, B. A. Eales, F. F. Cai, B. J. Read, 與 J. V. Healy。障礙選擇權定價：以神經網路建模。*Physica A: Statistical Mechanics and its Applications*, 344(1-2):289–293, 2004。

Y. Yang, Y. Zheng, 與 T. M. Hospedales。選擇權定價的閘控神經網路：設計上的理性。在 *Association for the Advancement of Artificial Intelligence*，頁 52–58, 2017。

J. Yao, Y. Li, 與 C. L. Tan。使用神經網路進行選擇權價格預測。*Omega*, 28(4):455–466, 2000。

T. Ye 與 L. Zhang。透過機器學習進行衍生性商品定價。SSRN 3352688, 2019。

C. Zapart。以小波分析與人工神經網路進行隨機波動率選擇權定價。*Quantitative Finance*, 2(6):487–495, 2002。

C. Zapart。超越 Black–Scholes：一種基於神經網路的選擇權定價方法。*International Journal of Theoretical and Applied Finance*, 6(05):469–489, 2003a。

C. Zapart。使用小波分析與人工神經網路的統計套利交易。在 *2003 IEEE International Conference on Computational Intelligence for Financial Engineering*，頁 429–435。IEEE, 2003b。

Y. Zheng。《機器學習與選擇權隱含資訊》。博士論文，Imperial College London, 2017。

Y. Zheng, Y. Yang, 與 B. Chen。隱含波動率曲面的閘控深度神經網路。arXiv:1904.12834, 2019。

31

W. Zhou, M. Yang, and L. Han. 以類神經網路進行可轉換公司債定價的非參數方法。In *Eighth ACIS International Conference on Software Engineering, Artificial Intelligence, Networking, and Parallel/Distributed Computing (SNPD 2007)*, volume 2, pages 564–569. IEEE, 2007.
