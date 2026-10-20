# FantasyWorld A/B/C — H200 部署與啟動手冊

適用 instance：56870338，專案 /workspace/fantasy-world。
部署版本：task/t_d38716c6-ccw-relay，HEAD 45dd5d260e8b78423e632a720d5fa155ad8a47f9。
本手冊不授權執行實驗。準備階段不載入大模型，也不執行生成；視覺／幾何效果仍 NOT_RUN。

### PM 驗收與文件發布邊界

PM 接受部署時點的證據作為「環境與離線資產已準備，待另行核准 model-load/inference」的交付，不代表目前 instance 仍可連線或資產仍保存。本次重算交付包 hash、19 個文字檔清單、15 項資產的數量／總大小／雜湊紀錄一致性與 5 個部署檔案 hash；沒有重新讀取遠端 140277466073 bytes 權重，也沒有重跑遠端測試。

發布前經唯一 wrapper 對 56870338 唯讀複查，回傳 exit 3：Selector must identify exactly one running instance; no fallback。故遠端現況無法重新確認；不推論它已銷毀或資料遺失，也不改用其他 instance、不啟動或操作平台。下列硬體、磁碟、測試與資產數字是部署時點紀錄，不是本次 live 驗證。

本 docs follow-up commit 只增加這份手冊；原始 evidence 與既有 h200-ab-plan／h200-c-plan 綁定部署 HEAD 45dd5d260e8b78423e632a720d5fa155ad8a47f9。即使只改文件，Git HEAD 也已改變，不能從新 commit 直接沿用舊 plan root 續跑。後續另獲推論批准時，優先在選定的新 HEAD 使用全新 run root 並重新 plan；若須使用舊 plan，必須保留原部署 commit 及完全相同的其餘 identity，不能手改 manifest 或 fingerprint。不要為了套用文件對遠端盲目 git pull 或切換 commit。

## 連線與保護

唯一 agent 入口（本機執行）：

```bash
/home/pierce/vast-ssh-workflow/vast-ssh --config /home/pierce/.config/vast-ssh/config.json --non-interactive --id 56870338 -- /usr/bin/id -un
```

使用者已批准 Vast 專用 accept-new TOFU；既存 key 變更仍 fail closed。
不改 hostkey、agent、driver、平台服務，不租用或 start/stop/rebuild/reboot/destroy。
所有安裝／資產僅在本專案／隔離環境；未複製本機 GH token 或其他專案資料。
/workspace 實測 workspace_is_volume=false：不是永久儲存；recycle/destroy 會失去資料。
不要以原始 vast-capabilities 全量輸出作交接，它可能包含管理面資訊。

wrapper 的遠端命令 timeout 為 120 秒。timeout 不等於 job 停止，不得盲目重播。
長實驗請由人透過可信互動入口開專用 tmux session，再執行下列命令；不要掛到既有研究 session。
若使用 canonical 非互動 wrapper，先由 PM 核准有唯一 PID、獨立日誌及 exit marker 的 detached bounded launcher，勿直接讓長 inference 受 120 秒限制。

## 環境與目前驗證

- NVIDIA H200 NVL，143771 MiB；driver 595.99.02，driver max CUDA 13.2。
- 已安裝 toolkit 12.8；torch wheel 自帶 CUDA 12.8。未動 driver/toolkit。
- 32 logical CPUs；/proc/meminfo MemTotal=263764500 kB；磁碟 quota=214748364800 bytes。
- isolated .venv：CPython 3.11.17、torch 2.7.1+cu128、torchvision 0.22.1+cu128、numpy 1.26.4。
- transformers 4.49.0、huggingface_hub 0.34.0、diffusers 0.39.0。
- requirements.txt、Python3.11 適用 requirements-relay-dev.txt、vendored thirdparty/utils3d 已安裝；完整 freeze 見 data/deployment/requirements-freeze.txt。
- 實測 CUDA available、H200 device、8-element CUDA tensor 平方和=140；Fusion、pose processor、MoGe、inference_wan22 import 與 CLI --help 通過。沒有模型載入。
- CPU suite 48 passed。A/B synthetic：5 calls，321/321 frames；C synthetic：4 calls，161/161/321 frames，resume 0 calls。都是 TEST_ONLY，不是生成樣本。
- 額外 pytest -W error：11 failed / 37 passed，仍為 imageio-ffmpeg 0.6.0 handle ResourceWarning／unraisable warnings。未修 site-packages、未隱藏警告。
- decord 0.6.0 官方 wheel 的 WHEEL metadata 是 cp36-cp36m-manylinux2010_x86_64；uv pip check 與 pip check 會報 platform incompatibility。Python3.11 實際 import 與 81-frame 32×32 CPU影片 decode 成功；不是所有 package checks 全綠。未修改 metadata。
- 大模型 VRAM 峰值、完整模型反序列化、推論品質、模型 ray/FoV 都尚未驗證。

日誌／證據在 data/deployment/：install.log/install.exit、imports-initial.log、pytest.log、pytest-strict.log、ab-synthetic.log、c-synthetic.log、decord-decode.log、pip-check*.log、download.log/download.exit、asset-verification.json、offline-preflight.log。
install.exit=1 是最後的 decord package check，不是 torch 安裝失敗；以後續 import/decode 證據解讀。

## 模型契約與離線使用

所有命令從專案根目錄執行；不可改 cwd。

```bash
cd /workspace/fantasy-world
source .venv/bin/activate
export HF_HOME=/workspace/fantasy-world/.hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
python data/deployment/preflight.py
```

不要把 --wan-ckpt-path 指到單一模型 repo；必須是 /workspace/fantasy-world/models。
實際 nested layout：

- models/PAI/Wan2.2-Fun-A14B-Control-Camera/{high_noise_model,low_noise_model}/diffusion_pytorch_model.safetensors（及各 config.json）
- models/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-{high,low}-noise-HPS2.1.safetensors
- models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/{high_noise_model,low_noise_model}.pth
- models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
- models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
- models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/{spiece.model,tokenizer.json,tokenizer_config.json,special_tokens_map.json}
- .hf-cache/hub/models--Ruicheng--moge-2-vitl-normal/ 的 model.pt snapshot 與 main ref

Wan2.1 namespace 只有 Wan2.2 loader 硬編碼 redirect 的共用 text/VAE/tokenizer，沒有下載 Wan2.1 DiT 或 GaussianPro 權重。
已完成下載與驗證：15 個檔案，共 140277466073 bytes（130.64 GiB）；LFS SHA256 全部匹配 HF 公布值，非 LFS config/tokenizer JSON 記錄 size 與實際 SHA256。逐項 repo revision/size/hash 見 assets-manifest.json 與 asset-verification.json；驗證後可用空間約 65.95 GB（61.42 GiB），執行前仍需重新檢查。
已實跑 offline preflight exit0；正式 model-path 的 h200-ab-plan 與 h200-c-plan 亦 exit0，狀態均 NOT_RUN。ModelConfig 仍印「downloading」，但 offline 時實際返回 existing local_dir；這不是關閉離線或網路補抓。
公開模型下載不需 login；offline preflight 只做 ModelConfig 路徑解析、tokenizer 編碼、cache lookup、imports／tiny CUDA，不初始化 FantasyWorldSampler 或 MoGeModel。
離線準備不等於整個 inference 路徑已實跑。若 runtime 仍要求下載，停止查缺檔，不要關閉 offline 來偷偷補。

下載是有界一次性任務，非推論／服務：PID 見 download.pid，輸出 download.log，完成以 download.exit=0 與 asset-verification.json 全項通過為準。
若中斷先查 PID/exit/log，不並行重跑。確認原 job 已退出後才可用 .venv/bin/python data/deployment/download.py 續下載（HF 保留 incomplete）；失敗 marker 必須保留歷史。不可刪 weights 或研究資料來騰空間。
SHA256 驗證僅串流讀檔；PTH 不反序列化，safetensors 只查 header。未把 pickle 權重當安全程式執行。

## 固定樣本與自備影像

固定輸入目錄：/workspace/fantasy-world/data/relay-inputs。
sample.png 是 repo examples/images/input_image.png（原592×336）的準備版，並非模型生成。
prompt.txt 保存正向文字，negative-prompt.txt 保存負向文字（目前空）；sample-provenance.json 記錄來源與 PNG hash。

規則：接受 PNG/JPG/JPEG；EXIF transpose → RGB → LANCZOS 等比縮放＋置中裁切至 832×480 → lossless PNG。
不拉伸、不以 MP4 解碼圖當 relay 端點。自備圖片的 resize/crop 會改視野；若需要校準 intrinsics，必須另外核對，不能由本規則推論 FoV 正確。
建議保留原檔於 inputs 內不同檔名，另產生準備 PNG；不覆寫現有 sample 或已用的 prompt。每張圖／prompt 變更用新 run root。
可用 data/deployment/prepare_image.py 原檔路徑 新PNG路徑；此工具拒絕覆寫。若換圖片，先複製並修改 relay.sh 的 --image 路徑或以該完整 argv 執行。

## A/B 與 C 的 plan / run

data/deployment/relay.sh 固定所有參數：90°、radius=1（數值單位非公尺）、focal=500、81 frames、832×480、16fps、50 steps、seed1024。
runner 自動明設 --using_scale False。A/B 前三段共用，A final/B final 條件不同，共五次 inference；C 為獨立四次 inference，不從 A/B 匯入。

僅規劃（已驗證，不載入模型）：

```bash
bash data/deployment/relay.sh ab plan output-relay/h200-ab-plan
bash data/deployment/relay.sh c plan output-relay/h200-c-plan
```

以下只在 PM／使用者另行批准完整推論後才執行；部署本身沒有執行。先檢查磁碟，debug.pth 與逐幀 PNG／pointcloud 可能很大；每次用新 root，別混入 synthetic 目錄。

```bash
# 專用互動 tmux session 內，已 cd/source/export
set -o pipefail
bash data/deployment/relay.sh ab run output-relay/ab-run-001 2>&1 | tee data/deployment/ab-run-001.log
bash data/deployment/relay.sh c run output-relay/c-run-001 2>&1 | tee data/deployment/c-run-001.log
```

不要把上面兩個 run 平行執行。plan 所選 root 若要直接 run，必須完全相同設定、source、HEAD、輸入與 checkpoint inventory；新 run-001 root 會自己建立 manifest。
每段日誌為各 segment 目錄下的 inference.log（stdout/stderr 合併）；runner stdout log 放 data/deployment。各段 raw first_frame.png/last_frame.png、video.mp4、receipt.json 和 debug.pth 也在各 segment 下。
A: output-relay/ab-run-001/A/segment_000..003；B: .../B/segment_003（前三段引用 A）。
C: output-relay/c-run-001/C/{L1,L2,R1,R2}。

中斷後重複完全相同的 run 命令／root 才是續跑。完成段須重新核對 receipt/hash/frames/fps；input/source/HEAD/weights inventory 變更會拒絕。不手改 manifest/fingerprint。
若 media 已寫出但未成功 commit manifest，runner 保守拒絕 partial artifacts；保留原目錄查因，換新 root，不自動覆蓋或偽造 COMPLETE。
不要在 planned/in-progress run 中下載、替換模型，因 inventory 包括 nested 路徑的 size/mtime。

## Stitch 與輸出

manifest.stitch_commands 是 authoritative argv；完成 inference 後才能拼接：

```bash
python data/deployment/stitch.py output-relay/ab-run-001/manifest.json
python data/deployment/stitch.py output-relay/c-run-001/manifest.json
```

helper 僅接受 backend=gpu、status=COMPLETE，依序 subprocess.run(argv, check=True)，不 shell-eval。已存在輸出不覆寫。
A/B 輸出 A/joined.mp4、B/joined.mp4，預設各321 frames。
C 輸出 C/CCW.mp4、C/CW.mp4 各161 frames，C/tour.mp4 321 frames。
C tour=L1,L2,reverse(R2),reverse(R1)，反段序也反幀序；CW 動態時間倒播，不是整圈 forward-time 動態影片。
拼接重新編碼且只丟掉每後段第一個重疊時間樣本；不是接縫修補或 lossless copy。

## 必須保留的限制

ray_policy=legacy_pixel_K_rescaled_again_UNVERIFIED_FOV：pixel K 在 direct ray path 再乘 W/H，principal point 亦有半像素差異。CPU global c2w 閉合不證明真實模型 FoV／Plücker 正確。
A/B/C 是 appearance-condition 實驗，不是已驗證幾何閉環；C 額外改變鏈深、方向及 end condition，不是純鏈深 ablation。
imageio strict warnings 與 decord metadata 診斷均未消除。GPU imports/CUDA arithmetic 只證明環境可用，不能冒稱完整模型、VRAM budget、視覺品質或閉環效果通過。
模型、輸入、cache、outputs、環境與部署日誌不入 Git。部署階段 HEAD 保持不變；PM 的 docs follow-up commit/push 只發布本手冊，Git HEAD 變更與舊 plan 相容性依本手冊開頭規則處理，不修改既有實驗 source。
