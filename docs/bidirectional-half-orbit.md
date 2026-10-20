# C：雙向半圈、指定方向條件式會合（CPU-tested prototype）

GPU imports / inference / visual or geometric quality: **NOT_RUN**。
這不是幾何閉環、不是對稱 teacher、也不是整圈 forward-time 動態影片。
沿用 [A/B 規約及環境](ccw-orbit-relay.md)，不改模型或 legacy ray-K 行為。

## 固定協定與資料流

| 段 | global 角度 | input | end_image | seed | 必須完成的依賴 |
|---|---|---|---|---|---|
| L1 | 0 → +90 | I0 | 無 | base | 無 |
| L2 | +90 → +180 | L1 raw last_frame.png | 無 | base+1 | L1 |
| R1 | 0 → −90 | I0 | 無 | base | 無 |
| R2 | −90 → −180 | R1 raw last_frame.png | L2 raw last_frame.png | base+1 | R1、L2 |

實作採簡單順序 L1、L2、R1、R2；雙第一段沒有平行化。只有 R2 有 end condition。
依賴 status、完整 receipt、影片實際幀數/fps、三份產物 hash 都通過驗證後才可使用。
R2 呼叫前即保存 input_hash、end_image_hash 與來源 ID；不是從壓縮 MP4 抽條件。
每段 generated raw uint8 首末 PNG 與影片都獨立保留，絕不以條件圖覆蓋輸出末幀。

這版刻意不支援跨 A/B 實驗匯入。C 獨立新 run 恰好四次 inference 後端呼叫；
同一 C manifest 可驗證後續跑、跳過已完成 prefix，但不會採用其他 root 的產物。
沒有 import 選項，不改寫或偽造舊 A/B manifest fingerprint，不把 TEST_ONLY 當 GPU 結果。

預設每段 90° / 81 frames。C 僅允許 --angle 90，其他角度明確拒絕；
frames 仍可選 >=5 的 4k+1，尺寸為正 16 倍數。seed 依每路深度設置，
兩路相同深度使用相同數字 seed，並不表示不同方向有相同噪聲傳播或模型效果。

## 相機、尺度與共同 K

OpenCV c2w：+X 右、+Y 下、+Z 前；world up = −Y。
CCW 定義是從 −world Y 看向 +Y，畫面右 +X、上 +Z。
使用同一 O=(0,0,r)、半徑 r、look-at／roll，L1/L2 直接沿用 A 的前兩段相機；
右路以 D @ global_c2w @ D（D=diag(−1,1,1,1)）產生負角度 proper rotations。
+180 與 −180 的完整 global c2w 在浮點誤差內相同，不只是位置一致。
每段保留 anchor_c2w=Gs 與 local_c2w，滿足 Gs @ local = global。
相機逐段首幀正規化是合法 gauge 轉換，不是遺失 global 累積路徑。

所有命令明設 --using_scale False。固定數值 r 不等於公尺，也不保證匹配訓練尺度；
不逐段由生成圖重估尺度。原 sampler 仍載入／執行 MoGe，沒有降低 GPU 需求的承諾。
所有段共用焦距與尺寸，loader K=[[f,0,(W−1)/2],[0,f,(H−1)/2],[0,0,1]]，
明列於 identity.loader_intrinsics；真實 camera loader 的 CPU 測試核對會合 K 與 pose。

A/B/C 共同 ray_policy 仍是 legacy_pixel_K_rescaled_again_UNVERIFIED_FOV：
既有 decoder 將主點改為 W/2,H/2，direct ray 路徑又將 pixel K 乘 W/H。
本次不修正此模型限制，沒有測 GPU tensor／Plücker 整合，數值會合不能證明模型 FoV 正確。

## CPU plan、測試、synthetic smoke

沿用現有 .venv；不重裝 requirements.txt、不下載模型。

```bash
.venv/bin/python bidirectional_relay.py --help
.venv/bin/python bidirectional_relay.py plan \
  --image examples/images/input_image.png --prompt 'A static room.' \
  --output output-relay/c-plan
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.bidirectional_cpu_smoke --output output-relay/c-cpu-smoke
```

每次 smoke 使用新的空 root。plan 寫 cameras／manifest，status=NOT_RUN；
smoke 使用 32×32 合成遞增灰階影片、四個後端呼叫，所有紀錄 TEST_ONLY，GPU NOT_RUN，
並驗證續跑 0 後端呼叫、torch 未載入，以及 CCW/CW 各 161、tour 321 個實際解碼影格。
這些不是模型場景或模型效果證據。

## 未執行的 GPU 命令介面

需另行核准 GPU 與完整離線資產後才可執行。套用 A/B 文件中的 GPU 環境與資產限制；
這版只預檢三個顯式 checkpoint 路徑，不代表 nested assets 齊全，模型 import 仍未驗證。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python bidirectional_relay.py run \
  --image /absolute/input.png --prompt 'A static room.' --output /absolute/new-c-root \
  --python /absolute/gpu-env/bin/python --wan-ckpt-path /absolute/Wan2.2 \
  --model-ckpt-high /absolute/FantasyWorld/high.pth \
  --model-ckpt-low /absolute/FantasyWorld/low.pth
```

run 產出四份原始 segment 產物；拼接是獨立 CPU 步驟，使用 manifest.stitch_commands 的
完整 argv（CCW、CW、tour 各一條；不直接以 shell 字串解析）。沒有 GPU 的本機只跑 plan/smoke。

## 影片展示、倒播與有界記憶體

manifest 明列三個 stitch_commands：

- CCW.mp4：L1 → L2（原始 CCW 支線）。
- CW.mp4：R1 → R2（原始 CW 支線）。
- tour.mp4：L1 → L2 → reverse(R2) → reverse(R1)。

四段 81 frames 時 tour = 321 frames；兩路各 161 frames。拼接保留首段全部樣本，
之後各段僅丟掉播放順序的第一個重疊時間樣本，沒有再移除整圈最後樣本。
既反段序、也反幀序；relay_media.py 的 --reverse 2 --reverse 3 是零起算 clip indices。
R2 倒播的 generated 最末幀會因接縫去重而被丟掉，但 raw PNG／原影片獨立保留，
沒有貼 L2 PNG 進 R2，也不表示相機同 pose 就有相同 generated RGB。

正播逐幀 decode/re-encode；倒播先逐幀寫 fixed-size RGB temporary disk spool，再反向 seek。
Python 活躍記憶體只持有固定少數影格，沒有 list(reader) 或 ffmpeg reverse 全段 RAM buffer。
每次只 spool 一個 clip，空間需 frames×H×W×3 bytes（外加 codec buffers），
tempfile 在輸出影片所在目錄，正常／例外離開都清理；需預留磁碟空間。
輸出不覆蓋既有檔；若拼接失敗留下 partial output，保留人工檢查，改用新輸出檔名。
此工具重新編碼，不是 lossless packet concat。

presentation.cw_time_reversed=true，並標示 CW 動態事件也倒播。靜態內容可作同向視角巡覽；
水、煙、人、光影等不會形成整圈向前時間。會合接縫、原圖回返可能不連續；
C 同時改變鏈深、方向與端點條件，不能聲稱純鏈深 ablation 或幾何一致性。

## 續跑與保護

相同命令／root 可續跑，file lock 防止併行寫，JSON 原子替換。以 config、I0 hash、
source hashes（含 C runner）、git HEAD、checkpoint path/size/mtime、完整 immutable plan
建立 fingerprint。執行時重建 plan，拒絕改過的 command、依賴、condition 路由與 camera hash。
權重 inventory 不是完整權重內容 hash；不可原地換同 size/mtime 的權重再宣稱同一性。

已完成段必須重新驗證 receipt、endpoint/video hashes、影片實際 frame count/fps。
壞檔 fail closed，不自行重生覆蓋。失敗且未留下 media 的段可重試；寫完 media 卻未 commit
manifest 的中斷窗口保守拒絕自動採用，請保留並用新 root。非空無 manifest root 拒絕覆寫。
TEST_ONLY 與 GPU backend fingerprint／執行入口分開，無法把測試產物當模型成功。

A/B 程式與舊研究產物未改寫；shared media source 有增量修改，因此舊 manifest 若記錄的是
舊 relay_media.py hash，現版會依既有嚴格 source policy 拒絕續跑。使用保存的原版程式或新 root，
不要手改舊 fingerprint。這是有意保留的來源驗證，不是自動資料遷移。

已知 imageio-ffmpeg 0.6.0 subprocess handles ResourceWarning 仍未解決，未修改 site-packages
或抑制 warnings。正常 pytest 通過；額外 -W error 仍失敗，完整結果見 C 交付報告。
