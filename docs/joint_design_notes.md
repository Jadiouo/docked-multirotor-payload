# 模組化多旋翼接頭設計查證筆記

查詢日期：2026-10-25（下文「查詢日期」皆同）。
標記規則：【原文】逐字引句（英文原文，來自 PDF 抽出文字）；【推論】為我的推算，不是出處明說；「找不到」表示查過仍無。
PDF 的抽字有些排版瑕疵（例如把 1/8 inch 寫成「1/800」），已在條目中註明。論文數字請在寫入論文前再對 PDF 原圖核一次。

---

## 1. ModQuad 與 H-ModQuad

### 1.1 ModQuad（Saldaña, Gabrich, Li, Yim, Kumar, ICRA 2018）
來源：https://www.lehigh.edu/~das819/pdf/icra18-modquad.pdf（查詢日期 2026-10-25，已下載全文）

| 項目 | 數據 | 原文 |
|---|---|---|
| 飛行器 | Crazyflie 2.0，27 g，最大酬載 15 g | "The vehicle weighs 27g, having a maximum payload of 15g. Its dimensions are 92x92x29mm" |
| 框架尺寸 | 116 x 116 x 48 mm | "The dimensions of this cuboid are 116x116x48mm." |
| 框架材料 | 碳纖維桿 + 8 個 3D 列印 ABS 接頭 | "Light-weight carbon fiber rods connected by eight 3-D printed ABS connectors form the frame." |
| 框架質量 | 7 g | "Our current frame design weights 7g , about half the payload capability." |
| 單一模組質量 | 40 g（含 Crazyflie、框架、對接機構） | "...strong force in comparison to the module mass (40g)" |
| 磁鐵 | NdFeB 方形片 6.35 x 6.35 x 0.79 mm，每個 ABS 接頭 2 顆，共 8 個接頭 | "The magnets are ... square, 6.35x6.35x0.79mm. Two magnets are located at each of the eight 3-D printed connectors" |
| 對接強度 | 面對面 4 顆磁鐵合計等效 1 kg | "When two modules are connected face-to-face, the four magnets are able to provide a bonding force equivalent to a 1kg" |
| 強度重量比 | 約 25:1 | "The strength-to-weight ratio is approximately 25:1." |
| 脫開 | 未做，留待未來 | "Once attached, undocking is a difficult task and is left to future work." |

備註：
- 論文自己說明模組只做水平面（垂直面）對接，上下面不對接。
- 論文未報告剪力、彎矩或對接失敗率。找不到（查過全文 grep：fail、detach、shear、strong）。
- 另一個版本的 ModQuad 模組（RAL 2019 自我拆解版，見 1.3）是 36.7 g，兩篇的「單一模組質量」不同，引用時要標明是哪一代。

### 1.2 H-ModQuad（Xu, D'Antonio, Saldaña, arXiv 2202.00788v3）
來源：https://arxiv.org/abs/2202.00788 與 https://arxiv.org/pdf/2202.00788（查詢日期 2026-10-25，已讀全文）

- 單一模組質量與酬載：【原文】"Each module has four brushless motors and a Crazyflie Bolt control board, weighing 135 g including a 2-cell LiPo battery, with a payload capability of 128 g."（這是 Crazyflie Bolt 無刷版，不是 30 g 的 Crazyflie 2.1，尺寸不能直接拿來比。）
- 對接機構：【原文】"The docking mechanism consists of permanent magnets mounted at the 3D-printed module frame corners. The disc-shaped lightweight Neodymium magnets create rigid connection between modules. Our experiments assume the docking is completed and focus on the behaviors of the assembled structures."
- 框架尺寸、磁鐵型號與數量、接頭強度、脫開、對接失敗：找不到（全文 grep：magnet、dock、detach、fail 皆無數據）。論文只指向外部組裝文件 https://docs-quad.readthedocs.io/en/latest/assemble/design.html（我沒有讀此頁，未驗證）。
- 論文把整個結構當成單一剛體處理（"A structure is a set of n >= 1 rigidly connected..."），沒有接頭內力分析。

### 1.3 補充：同一團隊的其他硬體論文
**ModQuad 自我拆解（Saldaña, Gupta, Kumar, RA-L 2019）** https://www.lehigh.edu/~das819/pdf/ral19_self_disassembly.pdf（查詢日期 2026-10-25）
- 框架與模組質量：【原文】"we reduced the weight of the frame from 8.5 g to 6.7 g with final dimensions of 157 mm x 157 mm. The total weight of our module including the modular frame is 36.7 g."
- 碳纖維桿徑：【原文】"The edges of the square are carbon fiber rods with a diameter of 1.37 mm" （1.37 mm 是抽字結果，與 ICRA 2018 的敘述風格不同，需對原圖確認）
- 接頭：【原文】"The dimensions of the connector are 29.17 mm x 9.55 mm and it weighs 0.9 g. It has a matrix of 3 x 10 holes where Neodymium Iron Boron (NdFeB) magnets can be placed. The magnets have 1.6 mm diameter and 0.8 mm thickness."
- 剪力：【原文】"Rotation about the roll axis Y[1] is an optimal way to separate a pair of modules since the shear force of the magnets is only 30% of the pull force."
- 設計意圖：【原文】"The connection is sufficiently weak to allow a pair of robots to detach when opposite maximum..."（句子在抽字處被截斷，需看原 PDF）。這代表此設計「刻意」讓接頭弱到馬達可以拆開，和需要強接頭的設計目標相反。

**ModQuad-Vi（Li, Gabrich, Saldaña, Das, Kumar, Yim, ICRA 2019）** https://www.lehigh.edu/~das819/pdf/icra19-modquadvi.pdf（查詢日期 2026-10-25）
- 框架：【原文】"The dimension of the frame is 320 mm x 320 mm x 112 mm and the weight is 160 g." 材料："composed of 20 mm diameter carbon fiber tubes and 3-D printed ABS connectors"。
- 飛行器：自製四旋翼，"platform weighs 971 g"，每個轉子最大推力 1024 g（大型機，不是 Crazyflie 尺寸）。
- 磁鐵：【原文】"square Neodymium (NdFeN) magnets with dimensions of 12.7 mm x 12.7 mm x 3.175 mm. The ABS connectors at each vertex contains two magnets. When two modules are docked with 4 pairs, the normal force supports 16.48 kg. This magnetic field corrects misalignment errors up to 1 cm" （原文 "NdFeN" 應為筆誤；16.48 kg 是論文自己的數字，無量測方法說明）

**Flying Gripper（ICRA 2018 ModQuad gripper）** https://www.lehigh.edu/~das819/pdf/icra18-modquad-gripper.pdf（查詢日期 2026-10-25）
- 籠子 8 g、Crazyflie 27 g。圓柱磁鐵質量 0.377 g。
- 【原文】"The cylindrical magnets have mass 0.377g and are able to generate a force of 0.4 kg in a tangential connection between two of the same magnets." "Note that the connections are not rigid - each forms a one DOF hinge."
- 磁鐵尺寸抽成 "1/800 of diameter and 1/400 of thickness"，應為 1/8 inch 直徑、1/4 inch 厚（【推論】：3.175 mm x 6.35 mm 圓柱以 NdFeB 密度約 7.5 g/cm3 算約 0.375 g，與 0.377 g 吻合）。
- 這是邊對邊（edge-to-edge）磁鐵接合，形成單自由度鉸鏈，不是剛性。

---

## 2. MARS-Dragonfly 與其他有磁吸對接的模組化飛行器

### 2.1 MARS-Dragonfly（Huang et al., arXiv 2604.05499）
來源：https://arxiv.org/abs/2604.05499 、https://arxiv.org/pdf/2604.05499（查詢日期 2026-10-25，已讀全文）
- 機體：【原文】"we used a quadrotor with a 380-mm wheelbase and a carbon-fiber unit frame... the total mass was approximately 1.55 kg."（單元 1.55 kg，不是 Crazyflie 尺寸。）
- 磁吸力：【原文】"With the aid of the attractive magnetic force (approximately 100 N), the docking mechanism accommodates rotational errors of up to 12.5 deg in roll, 15.5 deg in yaw, and 10 deg in pitch" ；位置容許 "x- and y-axis position errors up to +-35 mm"。
- 機構：磁鐵陣列被動對接、被動機械鎖（twist-lock）、單顆微型伺服（20 g）轉動公頭切換磁極做排斥分離。【原文】"our separation is driven by a 20 g mini servo motor"。
- 鎖定強度：做了 135 度擺錘測試（Fig. 11(e)）："A 135 deg pendulum test demonstrates the strong mechanical strength of the docked connection."，但沒有給力或力矩數值。找不到數值。
- 失敗模式：論文有明確記載分離失敗：【原文】"uncertain contact conditions may cause the male connector to lock inside the female mechanism"；沒有排斥力時分離延遲約 2 s（Fig. 15）。
- 對 ModQuad 類純磁鐵設計的評語（可引用為「相關工作批評」）：【原文】"the tangential holding force is insufficient for stable lateral support" ；"pure magnetic connections [12], [13] might break under disturbances or unbalanced payloads."
- 對接碰撞：對接瞬間懸停機 z 向加速度峰值約 12.63 m/s^2（Fig. 13 內文）。

### 2.2 LEGION（Sugihara et al., arXiv 2605.19431v1）
來源：https://arxiv.org/pdf/2605.19431（查詢日期 2026-10-25，已讀全文）
- 磁拉力最大 265 N（可切換磁鐵）；【原文】"The load test showed the maximum tensile tolerance of 300 N, reflecting the sufficient connection strength compared with the maximum force of 60 N generated by a single LEGION module."
- 模組主體 3.9 kg（大型機）。
- 有做接頭受力估測與控制，見第 3 節。

### 2.3 BEATLE（Sugihara et al., arXiv 2404.09153）
來源：https://arxiv.org/pdf/2404.09153（查詢日期 2026-10-25，已讀全文）
- 【原文】"a remaining issue with magnetic connections is the lack of resistance against bending moments." 因此加上機械鎖。
- 可切換磁鐵拉力 220 N；主體質量 4.2 kg。
- 另說明小磁鐵接頭可靠「瞬間彎矩」脫開：【原文】"...equipped with a much smaller magnetic connector, can perform aerial undocking motion by applying a momentary bending moment to the docking part with rotor thrusts, this smaller connector conversely complicates the docking motion."

### 2.4 查不到的項目
- 「Fly-Crane」：搜尋只找到 G-Fly-Crane（Jimenez-Cano 等，纜繩懸吊多機，不是磁吸對接）。沒有找到磁吸版的 Fly-Crane。找不到。
- 「Gandhi/Saldaña 後續工作」：搜到 Litman, Gandhi, Phan, Saldaña, RA-L 2021（視覺自組裝，偏演算法）與 ICRA 2021 ModQuad 論文 https://www.cis.upenn.edu/~linhphan/papers/icra21-modquad.pdf，我沒有讀全文，未確認硬體與接頭數據。找不到（未驗證）。
- 另有 TRADY（arXiv 2303.07106）搜尋結果提到可切換磁鐵，未讀，不採用。

---

## 3. 有沒有分析對接後接頭內力、彎矩或因內力脫開

有，但都不是針對 Crazyflie 尺寸、也沒有給接頭受力上限的設計值：

1. **Saldaña, Gupta, Kumar, RA-L 2019（自我拆解）**：把磁力與馬達力矩放在同一個力矩平衡式。模型：以旋轉點為軸，兩個磁力 fm1、fm2 與馬達推力產生的力矩 tau_max 競爭；最大可用力矩 tau_max = 2d(f_min - f_max)（式 8），設計條件 |tau_max| > |tau_m|（式 10），並推導 |fm1| < (tau_max - z_m2 * fm2)/z_m1 的上限（式 11 附近，抽字不完整）。用途是「讓磁力小到馬達能拆開」，不是承載分析。https://www.lehigh.edu/~das819/pdf/ral19_self_disassembly.pdf
2. **LEGION（arXiv 2605.19431）**：明確指出前人把結構當單一剛體、忽略接頭接觸力矩：【原文】"previous work [31, 41, 42] still ignores the interaction force and torque—i.e., the contact wrench—between interlocked modules during flight". 模型：第 i 個模組外力 w_i^ext = w_{i-1}^ctc - w_i^ctc + w^dist（式 8），把相鄰模組對接面的 contact wrench 當作線性系統估計並控制；超過「allowable wrench threshold」會判定對接成功／失敗。報告量測值：對接接觸力矩約 0.13 Nm 與 0.5 Nm（approach speed 0.3 m/s，大型機）。
3. **MARS-Dragonfly**：建立「虛擬四旋翼」的合力／合力矩可行集，沒有算接頭內力；但有討論意外脫開風險（見 2.1）。
4. **BEATLE**：定性指出磁接頭抗彎矩弱。

結論：針對 Crazyflie 尺寸磁接頭的「內力／彎矩上限」分析，我沒找到。找不到。接頭受力上限需要自己用力矩平衡推導（見第 6 節，標為推論）。

---

## 4. 小型 N52 圓片磁鐵規格

### 4.1 K&J Magnetics
- D42-N52（1/4 in x 1/8 in）https://www.kjmagnetics.com/proddetail.asp?prod=D42-N52 （2026-10-25）
  - 【原文】"6.35 mm diameter x 3.17 mm thick"；Pull Force case 1 "2.85 lb (1.29 kg)"（吸附厚鋼板）；case 2 "4.59 lb (2.08 kg)"；Surface field "5233 Gauss"；"0.0266 oz (0.75 g)"；最高溫 "176 F (80 C)"。
- D62-N52（3/8 in x 1/8 in）https://www.kjmagnetics.com/proddetail.asp?prod=D62-N52 （2026-10-25）
  - 【原文】"9.52 mm diameter x 3.17 mm thick"；case 1 "5.46 lb (2.48 kg)"；case 2 "11.27 lb (5.11 kg)"；"4105 Gauss"；"0.0599 oz (1.7 g)"。
- 注意：K&J 的 Pull Force 是「吸附到平整厚鋼板」的測值，不是磁鐵對磁鐵。K&J 說明：測試鋼板 "very flat, raw steel plates, held quite parallel by our test fixture. It's an ideal situation." 氣隙會大降：D82 在約 0.03 in 氣隙時 "The force drops to 3 lb"（原為 6.44 lb）。（來源：https://www.kjmagnetics.com/blog/how-much-will-a-magnet-hold ，2026-10-25）
- 磁鐵對磁鐵：同文章 "Two magnets attracting each other keep 'about 50% of the listed pull force'"（此為「拉力」相對於標示值的比例，不是剪力）。

### 4.2 supermagnete（Webcraft）
- S-09-03-N52N（9 mm x 3 mm，N52）https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-9mm-3mm_S-09-03-N52N （2026-10-25）：重 "1,5 g"；holding force "approx. 2 kg (approx. 19,6 N)"；displacement force "approx. 400 g (approx. 3,92 N)"。剪/拉 = 20%。
- S-08-06-N52N（8 mm x 6 mm，N52）https://www.supermagnete.nl/eng/disc-magnets-neodymium/disc-magnet-8mm-6mm_S-08-06-N52N （2026-10-25）：重 "2,3 g"；holding force "approx. 2,6 kg (approx. 25,5 N)"；displacement force "approx. 510 g (approx. 5,04 N)"。剪/拉 = 19.6%。
- 6 mm 規格的 N52 圓片 supermagnete 頁面我沒有找到。

### 4.3 法向吸力 vs 剪力：這個「15–25%」說法的查證
- 成立（磁鐵貼鋼板、靠摩擦）：supermagnete FAQ，https://www.supermagnete.hu/eng/faq/What-is-the-difference-between-adhesive-force-and-displacement-force-shear-force （2026-10-25）：【原文】"its maximum value is much smaller (approx. 15-25% of the specified adhesive force)." 並說明用材料選擇可 "increased to up to 50 % of the magnetic adhesive force."
- 上面兩個 supermagnete 商品頁的實測比約 20%，與此一致。
- Radial Magnet 指南（https://radialmagnet.com/magnets-for-holding-and-latching/，2026-10-25）：【原文】"it is typically just 15–25% of the rated pull"（二手廠商，只當佐證）。
- K&J：磁鐵貼鋼板 "you might only get 10% - 25% of the listed pull force when loaded this way"（摩擦係數 0.1 至 0.25）。
- **重要反例（會影響你的接頭模型）：磁鐵對磁鐵的剪力不是 15–25%。**
  - K&J 的剪力實驗與模擬（兩塊磁鐵互相滑動）：【原文】"we've always estimated the shear force at around half of the attraction force." 實測比 47% 至 70%，模擬 40–75%，峰值在偏移約 1/2 in 處。（https://www.kjmagnetics.com/blog/magnet-shear-force ，2026-10-25）
  - ModQuad 自我拆解論文：磁鐵陣列對陣列的剪力為拉力的 30%（見 1.3）。
  - Flying Gripper 論文：同款圓柱磁鐵「切向」連接 0.4 kg，無法與其軸向拉力對照（論文沒給）。
  - 【推論】兩片磁鐵面對面接合的 CAD 設計，剪力比建議用 0.3（論文值）當設計值，0.15 當保守下限，不要只用 0.15–0.25 這個「磁鐵貼鋼板」數字；若接頭有幾何卡榫（如 MARS-Dragonfly 的錐形與倒鉤），剪力改由結構承受，磁鐵只負責吸合。

---

## 5. Crazyflie 2.1 官方數據

來源：Bitcraze 官方 Datasheet Crazyflie 2.1 Rev 3，https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf （2026-10-25，已讀全文）
- 質量：【原文】"Takeoff weight: 29g"；開頭 "only weighs 29g"。
- 尺寸：【原文】"Size (WxHxD): 92x92x29mm (motor-to-motor and including motor mount feet)"
- 建議酬載：【原文】"Max recommended payload weight: 15 g"
- 飛行時間：【原文】"Flight time with stock battery: 7 minutes"
- 馬達軸距：datasheet 只給 92 x 92 mm 外形（含馬達座腳）。單獨的「對角馬達中心距」數字，我沒有在 Bitcraze 官方文件找到。找不到。
  【推論】92 mm 為含腳的外形，馬達中心距會略小；若要精確的軸距，請另查 Bitcraze 的 CAD/STEP 檔或韌體 simulation 參數（我沒查）。
- 最大推力：Bitcraze 推力升級套件文章（https://www.bitcraze.io/2022/10/thrust-upgrade-kit-for-the-crazyflie-2-1/ ，2026-10-25）：【原文】"the stock Crazyflie 2.1 with a 60 grams of total maximum thrust."；升級後 "~80 grams"。
  - 另一份 Bitcraze 頁面（pwm-to-thrust，https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/ ）列出 "at 93.75% PWM the trust is about 58g"（2015 年量測，全機）；與 60 g 一致。
  - 每顆馬達約 15 g 是我除以 4 的【推論】。
- 注意：ModQuad 論文以 Crazyflie 2.0 為基礎（27 g）。2.1 為 29 g；模組質量不能直接套用。

---

## 6. 建議給 CAD 和模擬用的數值

| 項目 | 建議值 | 來源類型 | 備註 |
|---|---|---|---|
| Crazyflie 2.1 質量 | 29 g | 官方規格 | Bitcraze datasheet |
| Crazyflie 外形（含腳） | 92 x 92 x 29 mm | 官方規格 | 非精確軸距 |
| 建議酬載 | 15 g | 官方規格 | 框架 + 磁鐵 + 電子零件要在內 |
| 全機最大推力 | 約 60 g（約 0.59 N） | 官方（Bitcraze 部落格） | 每顆約 15 g 為推論 |
| ModQuad 框架（參考值） | 116 x 116 x 48 mm，7 g，碳纖維桿 + ABS 接頭 | 論文 | ICRA 2018 |
| 單一模組質量（含框架與磁鐵） | 36.7 至 40 g | 論文 | RAL 2019 / ICRA 2018 |
| 磁鐵（ModQuad 參考） | 6.35 x 6.35 x 0.79 mm 方片，每接頭 2 顆 | 論文 | 面對面 4 顆等效 1 kg |
| 候選磁鐵 D42-N52 | 6.35 mm x 3.17 mm，0.75 g，吸鋼板 1.29 kg | 官方規格（K&J） | 是對鋼板，非對磁鐵 |
| 候選磁鐵 S-09-03-N52N | 9 x 3 mm，1.5 g，拉 2 kg，剪 0.4 kg | 官方規格（supermagnete） | 剪/拉 = 20% |
| 單顆磁鐵對磁鐵拉力 | 取標示值的 50% 為保守設計值 | 官方（K&J 部落格） | 需實測 |
| 剪力/拉力比（磁鐵對磁鐵，設計值） | 0.3（上限）、0.15（保守下限） | 論文（0.3）＋推論（0.15） | 不要用 0.15–0.25 當磁對磁上限，K&J 說約 0.5 |
| 剪力/拉力比（磁鐵對鋼板） | 0.15–0.25 | 官方規格（supermagnete、K&J） | |
| 接頭拉力上限（單接頭，Crazyflie 級） | 以 D42-N52 等級，單顆約 0.65 kgf（取 50%），接頭 2 顆約 1.3 kgf（約 12.7 N） | 推論 | 可對照 ModQuad「4 顆 = 1 kg」同量級；務必實測 |
| 接頭剪力上限 | 拉力上限 x 0.3，約 3.8 N | 推論 | 同上，實測驗證 |
| 彎矩（脫開）上限 | 以 RAL 2019 式 8 與 10：tau_max = 2d(f_min - f_max) 對比磁力矩 | 論文（模型）＋推論（數值） | 數值需用實際 d 與推力代入；我未計算 |
| 模擬的脫開判據 | 對每個接頭計算法向力、剪力、彎矩，任一超過上限即脫開 | 推論 | 模型參考 LEGION 的 contact wrench 做法（論文） |

未解決／需要你決定：
1. Crazyflie 官方的精確馬達中心距：找不到，建議看 Bitcraze CAD。
2. 接頭實際強度要靠實測（拉力、剪力、彎矩各一次），文獻沒有 30 g 級磁接頭的設計值。
3. 論文引用前，上列 PDF 抽字（尤其 1.37 mm 碳桿、ModQuad-Vi 的 NdFeN 與 16.48 kg）請對照原 PDF 圖面確認。

---

## 附：實際查過的地方
arXiv 2202.00788、2604.05499、2605.19431、2404.09153（全文）；Lehigh 主機上的 ICRA 2018 ModQuad、ICRA 2018 gripper、ICRA 2019 ModQuad-Vi、RA-L 2019 自我拆解（全文）；Bitcraze 產品頁（無規格）、datasheet、推力升級部落格、pwm-to-thrust 文件；K&J Magnetics 商品頁與兩篇部落格；supermagnete 商品頁與 FAQ；Radial Magnet 指南。
未讀：ICRA 2021 ModQuad、RA-L 2021 Litman 等、TRADY、H-ModQuad 外部組裝文件。
