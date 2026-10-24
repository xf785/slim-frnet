# SLIM-FRNet 路 鍚堟垚鎸囩汗鏁版嵁闆嗗钩鍙?
[![CI](https://github.com/xf785/slim-frnet/actions/workflows/ci.yml/badge.svg)](https://github.com/xf785/slim-frnet/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-912%20passed-brightgreen.svg)](tests/)

> 鐗堟湰 **v2.0** 锝?GUI 浼樺厛 锝?4 涓牳蹇冧緷璧?锝?瀹屽叏鍙鐜?锝?MIT
> 鎸?[璁捐鏂囨。](docs/DESIGN.md) 瀵瑰師椤圭洰鍋氭灦鏋勭骇閲嶅缓锛氫繚鐣欏凡楠岃瘉鐨勭敓鎴愬唴鏍革紝閲嶅啓閰嶇疆銆佹牳蹇冨煙灞傘€佹彃浠跺眰銆佹湇鍔″眰涓庡浘褰㈢晫闈€?
涓€閿敓鎴?*瀹屽叏鍙鐜?*鐨勫悎鎴愭寚绾规暟鎹泦锛氫富鎸囩汗 鈫?鎹哄嵃锛堝舰鍙?/ 浼犳劅鍣?/ 閫€鍖栭摼锛夆啋 TTA锛?杈撳嚭鍥惧儚銆佹帺妯°€佹柟鍚戝満銆佺粏鑺傜偣銆佽川閲忓垎銆佸垝鍒嗕笌鏍￠獙鍜岋紝闄?8 椤靛浘褰㈢晫闈€?
<!-- 鐣岄潰鎴浘 -->
| 鐢熸垚鐩戞帶 | 缁撴灉棰勮 |
|---|---|
| ![鐢熸垚鐩戞帶](docs/screenshots/gui-01-monitor.png) | ![缁撴灉棰勮](docs/screenshots/gui-02-preview.png) |
| **鍗虫椂娓叉煋锛堥€愮畻瀛愬姣旓級** | **TTA 閰嶇疆** |
| ![鍗虫椂娓叉煋](docs/screenshots/gui-03-live-render.png) | ![TTA 閰嶇疆](docs/screenshots/gui-04-tta.png) |

---

## 鈿狅笍 鍚堟垚鏁版嵁澹版槑

鏈蒋浠剁敓鎴愮殑鍥惧儚**瀹屽叏鐢辩畻娉曞悎鎴?*锛屼笉鍚换浣曠湡瀹炰汉绫绘寚绾规暟鎹紝涓嶆秹鍙婁釜浜虹敓鐗╃壒寰佷俊鎭€?鐢熸垚绠楁硶澶嶅埢鑷?SFinGe 鏂规硶瀛︼紙Cappelli et al., *Synthetic Fingerprint-Image Generation*, ICPR 2000锛夈€?鍙敤浜庣畻娉曠爺绌躲€佹ā鍨嬮璁粌銆佹暟鎹寮轰笌鏁欏锛?*涓嶅緱**鐢ㄤ簬鍐掑厖鐪熷疄韬唤鎴栬閬跨敓鐗╄瘑鍒郴缁熴€?
---

## 1. 30 绉掍笂鎵?
**Windows 鐢ㄦ埛鏈€鐪佷簨**锛氬弻鍑?[start_gui.bat](start_gui.bat) 鍗冲彲锛堣嚜鍔ㄦ壘 Python銆佹鏌ヤ緷璧栥€佸惎鍔ㄧ晫闈級銆?鍛戒护琛屽悓鐞嗭紝鍙屽嚮 [start_cli.bat](start_cli.bat)锛屾垨 `start_cli.bat selfcheck`銆?
鎵嬪姩鏂瑰紡锛?
```bash
# 1) 瀹夎锛堟牳蹇冨彧鏈?4 涓緷璧?+ 涓€涓?Qt 缁戝畾锛?pip install -r requirements.txt

# 2) 鑷锛氫笉闇€瑕佷换浣曞閮ㄦ暟鎹紝楠岃瘉鏁存潯閾捐矾
python -m synthfp selfcheck

# 3) 鐢熸垚涓€涓皬鏁版嵁闆嗭紙绾?30 绉掞級
python -m synthfp generate --preset quick_test

# 4) 鍚姩鍥惧舰鐣岄潰锛堟帹鑽愬叆鍙ｏ級
python -m synthfp gui
```

> GUI 闇€瑕?PySide6 **鎴?* PyQt6锛岃涓€涓氨澶熴€傜▼搴忎紭鍏堢敤 PySide6锛?> 鎵句笉鍒版椂鑷姩鍥為€€ PyQt6 鈥斺€?宸茬粡瑁呬簡 PyQt6 鐨勬満鍣ㄤ笉蹇呭啀瑁?PySide6銆?
鐢熸垚缁撴灉锛?
```
outputs/quick_test/
鈹溾攢鈹€ README.md                  # 鏁版嵁闆嗚鏄庯紙鍚堟垚澹版槑 + 瀛楁閲婁箟 + 澶嶇幇鏂规硶锛?鈹溾攢鈹€ dataset_manifest.jsonl     # 姣忚涓€涓牱鏈殑瀹屾暣璁板綍锛堝敮涓€鍏冩暟鎹潵婧愶級
鈹溾攢鈹€ plan.jsonl                 # 鐢熸垚璁″垝锛堝彲澶嶇幇渚濇嵁锛?鈹溾攢鈹€ config.resolved.yaml       # 瀹為檯鐢熸晥鐨勫畬鏁撮厤缃?鈹溾攢鈹€ environment.json           # 鐜蹇収
鈹溾攢鈹€ stats.json                 # 缁熻鎶ュ憡
鈹溾攢鈹€ splits/train.csv           # 鎸夎韩浠藉垝鍒嗭紙鏃犺韩浠芥硠婕忥級
鈹溾攢鈹€ checksums/sha256sums.txt   # 鍏ㄩ噺鏂囦欢鏍￠獙鍜?鈹溾攢鈹€ images/000001/left_thumb/imp_01/000001_left_thumb_imp01_tta00.png
鈹溾攢鈹€ mask/ orientation/ frequency/ centerline/ quality/ minutiae/
鈹斺攢鈹€ logs/{generation,done,failed}.jsonl
```

---

## 2. 瀹冨仛浠€涔?
涓€鏉″畬鏁寸殑鏍锋湰绠￠亾锛堣璁℃枃妗?4.4锛夛細

```
涓绘寚绾癸紙姣忎釜鎵嬫寚鍙畻涓€娆★級
  鎵嬫寚杞粨 5 鍙傛暟妯″瀷 鈫?Core/Delta 濂囧紓鐐?鈫?Zero-Pole 鏂瑰悜鍦?  鈫?澶氬眰 Perlin 棰戠巼鍥?鈫?Gabor 杩唬鑴婄嚎鍚堟垚
        鈹?        鈹溾攢鈹€ 鎹哄嵃 #1 鈹€鈹€ 闅忔満閫€鍖栭摼 鈫?鐨偆鎸夊帇鍙樺舰 鈫?浼犳劅鍣ㄧ壒鎬?鈹€鈹€鈹攢鈹€ TTA #0锛堝師濮嬪鐓э級
        鈹?                                                     鈹溾攢鈹€ TTA #1
        鈹?                                                     鈹斺攢鈹€ TTA #2 鈥?        鈹溾攢鈹€ 鎹哄嵃 #2 鈹€鈹€ 鈥︼紙鍚屼笂锛?        鈹斺攢鈹€ 鎹哄嵃 #N
                鈹?                鈹斺攢鈹€ 姣忎釜鏍锋湰锛歍TA 鈫?璐ㄩ噺璇勪及 鈫?缁嗚妭鐐规彁鍙?鈫?鏍囧噯鍖栧瓨鍌?鈫?Manifest
```

**涓轰粈涔堜互銆屼富鎸囩汗銆嶄负浠诲姟绮掑害**锛氫富鎸囩汗鍚堟垚鏄暣鏉￠摼閲屾渶璐电殑涓€姝ワ紙Gabor 杩唬锛夈€?鎶婂畠鏀惧湪鏉＄洰鍐呭鐢紝浣裤€屾瘡涓墜鎸囧彧绠椾竴娆°€嶆垚涓?*缁撴瀯鎬т繚璇?*鑰屼笉鏄紦瀛樼杩愭皵锛?澧炲姞鎹哄嵃鏁颁笌 TTA 鏁扮殑杈归檯鎴愭湰鍥犳鍙樺緱寰堜綆銆?
---

## 3. 鍥惧舰鐣岄潰锛堜竴绾у叆鍙ｏ級

鍙屽嚮 [start_gui.bat](start_gui.bat)锛屾垨 `python -m synthfp gui`銆?鎵撳紑鍚庢槸涓夊尯甯冨眬锛氬乏渚у鑸?锝?涓ぎ椤甸潰 锝?搴曢儴鐘舵€佹爮銆?
鎯冲厛鍦ㄦ病鏄剧ず鍣ㄧ殑鐜閲岀湅鐣岄潰闀夸粈涔堟牱锛圕I銆佽繙绋?shell锛夛紝鍙互绂诲睆鎴浘锛?
```bash
python tools/shoot_gui.py --dataset outputs/demo
# 鈫?outputs/gui-shots/01_鏁版嵁闆嗛厤缃?png 鈥?08_瀵煎嚭.png
```

| 椤甸潰 | 鍐呭 |
|------|------|
| **1. 鏁版嵁闆嗛厤缃?* | 韬唤鏁?/ 鎵嬫寚鏁?/ 鎹哄嵃鏁?/ TTA 鏁般€佽緭鍑虹洰褰曘€佺被鍒瘮渚嬶紙瀹炴椂褰掍竴鍖?+ 棰勮鏍锋湰鏁帮級銆佸垝鍒嗘瘮渚?|
| **2. 鍚庣閰嶇疆** | 鍚庣閫夋嫨銆佸垎杈ㄧ巼棰勮鎴栬嚜瀹氫箟灏哄銆佽剨绾垮悎鎴愬叏閮ㄥ弬鏁般€佸瑙傘€佹墜鎸囪疆寤撴瘮渚嬨€佸寮傜偣锛涗竴閿€屾覆鏌撴祴璇曞浘銆嶇湅鑰楁椂 |
| **3. TTA 閰嶇疆** | 21 涓畻瀛愬垎 6 绫诲嬀閫夈€侀€愪釜鍙傛暟婊戝潡 + 銆屽浐瀹氥€嶅紑鍏炽€佺粍鍚堢瓥鐣ワ紱**鏁堟灉瀵规瘮**椤靛疄鏃舵覆鏌撻€愮畻瀛愮粨鏋?|
| **4. 璐ㄩ噺涓庢爣娉?* | 璐ㄩ噺闃堝€间笌涓㈠純绛栫暐銆佺粏鑺傜偣鎻愬彇鍙傛暟銆両SO 妯℃澘銆佹爣娉ㄤ骇鐗╁紑鍏炽€佹嵑鍗伴樁娈碉紙鍙樺舰 / 浼犳劅鍣?/ 閫€鍖栭摼锛?|
| **5. 鐢熸垚鐩戞帶** | 杩涘害鏉°€? 涓疄鏃剁粺璁″崱鐗囥€佷换鍔¤〃锛堝惈鐘舵€佷笌璐ㄩ噺鍒嗭級銆佸僵鑹叉棩蹇椼€佹殏鍋?/ 鎭㈠ / 鍋滄銆佸け璐ョ瓫閫?|
| **6. 缁撴灉棰勮** | 鏁版嵁闆嗘爲娴忚 鈫?缂╃暐鍥剧綉鏍?鈫?澶у浘锛堢缉鏀?/ 骞崇Щ锛夛紝鍙彔鍔犳帺妯°€佺粏鑺傜偣銆佹柟鍚戝満銆佸寮傜偣锛涘彟鏈夈€屽嵆鏃舵覆鏌撱€嶉〉瀹炴椂璋冨弬 |
| **7. 缁熻鎶ュ憡** | 绫诲埆 / 璐ㄩ噺 / 缁嗚妭鐐?/ TTA 绠楀瓙 / 鍒掑垎 / 鎵嬫寚 鍏釜瑙嗚鐨勫垎甯冭〃锛屽潎鍙鍑?CSV |
| **8. 瀵煎嚭** | 閫愰」閫夋嫨瀵煎嚭鍐呭锛圧EADME / 缁熻 / 鏍￠獙鍜?/ 鍒掑垎 / 鐜锛夈€佸彲閫夋墦鍖?ZIP銆佹暟鎹泦缁撴瀯鎬昏銆佸鐜板懡浠や竴閿鍒?|

鐣岄潰缁嗚妭锛?
- GUI **涓嶇洿鎺ヨ皟鐢ㄤ换浣曠敓鎴愰€昏緫**锛屽彧璋冪敤搴旂敤鏈嶅姟灞傦紱鏈嶅姟灞傜敤鍥炶皟涓婃姤杩涘害锛?  鐢?`synthfp/gui/qt_bridge.py` 杞垚 Qt 淇″彿锛堣法绾跨▼闃熷垪鎶曢€掞級锛屽洜姝や富绾跨▼姘镐笉闃诲銆?- 鍙傛暟鏀瑰姩**鍗虫椂 Pydantic 鏍￠獙**锛氶潪娉曞€间笉浼氳惤鍦帮紝鎺т欢鍥炴粴骞剁粰鍑哄彲璇荤殑閿欒瀹氫綅銆?- 鏈崟鑾峰紓甯稿啓鍏?`crash.log` 骞跺脊绐楁彁绀猴紝涓嶄細闈欓粯宕╂簝锛堝師椤圭洰鏈€澶х殑鐥涚偣锛夈€?- 鏆楄壊 / 浜壊涓婚锛堣彍鍗曘€岃鍥?鈫?涓婚銆嶏級锛屼笉渚濊禆 qdarktheme 绛夐澶栧寘銆?
---

## 4. 鍛戒护琛岋紙楂樼骇鐢ㄦ埛锛?
```bash
python -m synthfp --help

# 鐢熸垚
python -m synthfp generate --preset quick_test
python -m synthfp generate -c configs/default.yaml --set runtime.workers=8
python -m synthfp generate --preset large_research --set dataset.identities=50 --dry-run

# 璁″垝棰勮锛堜笉鍐欎换浣曟枃浠讹級
python -m synthfp plan --preset default --json

# 鏁版嵁闆嗘搷浣?python -m synthfp stats    --dataset outputs/synth_fp_v1
python -m synthfp validate --dataset outputs/synth_fp_v1 --check-hash   # 閲嶇畻 SHA-256
python -m synthfp export   --dataset outputs/synth_fp_v1                 # 骞傜瓑閲嶅鍑?
# 閰嶇疆
python -m synthfp presets                                        # 鍒楀嚭 10 涓唴缃璁?python -m synthfp show --preset high_quality_only                # 鎽樿
python -m synthfp show --preset default --resolved               # 瀹屾暣 YAML

# 鍗曞紶棰勮锛堜笉钀界洏锛屽彲 --save锛?python -m synthfp preview --preset default --finger left_thumb --tta 1 --save out.png

# 鑷
python -m synthfp selfcheck

# 鍥惧舰鐣岄潰
python -m synthfp gui
```

`--set` 浣跨敤鐐瑰彿璺緞锛屽彲閲嶅锛屽€兼寜 YAML 瑙ｆ瀽锛?
```bash
python -m synthfp generate \
  --set dataset.identities=200 \
  --set dataset.tta_per_impression=5 \
  --set tta.mode=chain --set 'tta.chain_length=[2,3]' \
  --set 'tta.operators=[rotate,elastic,blur,jpeg]' \
  --set quality.min_score=55 --set quality.reject_on_fail=true \
  --set runtime.workers=12
```

---

## 5. 鍐呯疆棰勮

| 棰勮 | 瑙勬ā | 鐢ㄩ€?|
|------|------|------|
| `default` | 10脳2脳2脳3 = 120 | 鏍囧噯鏁版嵁闆?|
| `quick_test` | 2脳1脳1脳2 = 4 | 鐜鑷 |
| `large_research` | 500脳4脳3脳4 = 24000 | 绉戠爺澶ц妯￠泦 |
| `high_quality_only` | 120锛堥槇鍊?60 鍒嗕涪寮冿級 | 娓呮櫚搴︽晱鎰熶换鍔?|
| `robust_low_quality` | 120锛堥噸搴﹂€€鍖?+ 鍣０閬尅閾撅級 | 椴佹鎬?/ 鎷掕瘑鐮旂┒ |
| `class_balanced` | 50脳2脳2脳3 = 600 | 浜旂被鍚?20%锛屽垎绫讳换鍔?|
| `whorl_only` / `arch_only` / `loop_only` | 120 | 鍗曠被鍒爺绌?|
| `no_tta` | 10脳2脳2脳1 = 40 | 鍙骞插噣鎹哄嵃鍩哄簳 |

棰勮鏄?*瀵归粯璁ら厤缃殑瑕嗙洊鐗囨**鑰岄潪鐙珛 YAML 鍓湰锛屽洜姝?schema 鍗囩骇鏃堕璁捐嚜鍔ㄨ窡闅忋€佷笉浼氳厫鍖栥€?
---

## 6. 鍙鐜版€?
鍥涜绱犵己涓€涓嶅彲锛堣璁℃枃妗?0.2锛夛細

| 瑕佺礌 | 钀界洏浣嶇疆 |
|------|---------|
| **閰嶇疆** | `config.resolved.yaml` + manifest 姣忚鐨?`config_fingerprint` |
| **绉嶅瓙** | `runtime.seed` 娲剧敓鏁存５绉嶅瓙鏍戯紝姣忎釜鏍锋湰鐨勭瀛愬啓鍦?manifest 鐨?`seed` 瀛楁 |
| **鐗堟湰** | `environment.json`锛圥ython / 骞冲彴 / numpy / opencv / pydantic / synthfp 鐗堟湰锛?|
| **鏍￠獙鍜?* | `checksums/sha256sums.txt` + manifest 姣忚鐨?`sha256` |

绉嶅瓙娲剧敓鏄?*绾?SHA-256 灞傜骇鏍?*锛坄synthfp/core/seeds.py`锛夛紝涓嶄娇鐢?Python 鐨?`hash()`锛?
```
dataset 鈫?identity 鈫?finger 鈫?impression 鈫?tta
```

鍥犳锛氬悓涓€浠介厤缃?+ 鍚屼竴涓瀛愶紝鍦ㄤ换浣曟満鍣ㄣ€佷换浣曡繘绋嬨€佷换浣曠嚎绋嬫暟涓嬮兘寰楀埌**閫愬儚绱犵浉鍚?*鐨勭粨鏋溿€?`synthfp selfcheck` 浼氬疄娴嬭繖涓€鐐癸紙鍚岀瀛愪袱娆℃覆鏌撻€愬儚绱犳瘮瀵癸級銆?
---

## 7. 鐩綍缁撴瀯

```
synthfp/
鈹溾攢鈹€ config/          閰嶇疆绯荤粺锛歅ydantic schema + YAML 璇诲啓 + 10 涓璁?鈹溾攢鈹€ core/            鏍稿績鍩熷眰
鈹?  鈹溾攢鈹€ seeds.py         纭畾鎬х瀛愭爲锛堝彲澶嶇幇鍩虹煶锛?鈹?  鈹溾攢鈹€ types.py         鍏ㄩ儴棰嗗煙鏁版嵁缁撴瀯锛堝崟涓€鏁版嵁婧愶級
鈹?  鈹溾攢鈹€ planner.py       璁″垝灞曞紑 + 鎸夎韩浠藉垝鍒嗭紙闃叉硠婕忥級
鈹?  鈹溾攢鈹€ scheduler.py     浠诲姟璋冨害锛堢嚎绋嬫睜 / 涓茶銆佹柇鐐圭画浼犮€佸け璐ラ殧绂伙級
鈹?  鈹溾攢鈹€ manifest.py      鍞竴鍏冩暟鎹潵婧愶紙jsonl 杩藉姞鍐欙級
鈹?  鈹溾攢鈹€ storage.py       鐩綍甯冨眬 + 鍘熷瓙鍐?+ 鏍￠獙鍜?鈹?  鈹斺攢鈹€ quantize.py      鏂瑰悜鍦?棰戠巼鍥剧殑 uint8 绱у噾閲忓寲
鈹溾攢鈹€ pipeline/        鏍锋湰绠￠亾锛歝hain锛堝彲澶嶇幇绠楀瓙閾鹃€夋嫨锛? runner / stages
鈹溾攢鈹€ plugins/         鎻掍欢灞?鈹?  鈹溾攢鈹€ backends/        SFinGe 鍚庣锛堝鍒诲唴鏍革級銆丏ummy 鍚庣
鈹?  鈹溾攢鈹€ impressions/     鐨偆鎸夊帇鍙樺舰銆? 绉嶄紶鎰熷櫒鐗规€?鈹?  鈹溾攢鈹€ tta/             21 涓?TTA 绠楀瓙锛? 绫伙級
鈹?  鈹溾攢鈹€ quality/         鑷爺 NFIQ2 椋庢牸璐ㄩ噺璇勫垎
鈹?  鈹斺攢鈹€ minutiae/        鍚戦噺鍖?Zhang-Suen + Crossing Number + ISO 妯℃澘
鈹溾攢鈹€ app/             搴旂敤鏈嶅姟灞傦細GenerationService / PreviewService / exporter / stats
鈹溾攢鈹€ gui/             鍥惧舰鐣岄潰锛? 椤甸潰 + 涓婚 + 鎺т欢搴擄級
鈹?  鈹溾攢鈹€ qt.py            Qt 缁戝畾鍏煎灞傦紙PySide6 / PyQt6锛?鈹?  鈹斺攢鈹€ pages/           8 涓〉闈紝澹版槑寮忋€岄厤缃?鈫?鎺т欢銆嶅弻鍚戠粦瀹?鈹溾攢鈹€ legacy/          **鍘熼」鐩凡楠岃瘉鐨勭敓鎴愬唴鏍?*锛堜繚鐣欒祫浜э紝鏈€灏忔敼閫狅級
鈹?  鈹溾攢鈹€ render/          杞粨 / 濂囧紓鐐?/ 鏂瑰悜鍦?/ 棰戠巼鍥?/ Gabor 鑴婄嚎
鈹?  鈹溾攢鈹€ degrade/         19 涓寚绾归€€鍖栫畻瀛愶紙浣滀负鎹哄嵃鎵板姩澶嶇敤锛?鈹?  鈹斺攢鈹€ util/            Unicode 瀹夊叏鍥惧儚璇诲啓
鈹斺攢鈹€ cli.py           CLI锛堜粎鐢ㄦ爣鍑嗗簱 argparse锛?
start_gui.bat                 鍙屽嚮鍚姩鍥惧舰鐣岄潰锛堣嚜鍔ㄦ壘 Python銆佹鏌ヤ緷璧栵級
start_cli.bat                 鍙屽嚮鎵撳紑鍛戒护琛?/ 蹇€熸墽琛屽瓙鍛戒护
configs/default.yaml          榛樿閰嶇疆锛堜粠 schema 鐢熸垚锛?docs/DESIGN.md                閲嶅缓璁捐鏂囨。锛堟灦鏋勫喅绛栦笌鍙栬垗锛?docs/CONTRACTS.md             鎻掍欢灞傛帴鍙ｅ绾︼紙骞惰寮€鍙戠殑鍞竴绾﹀畾锛?docs/screenshots/             鐣岄潰鎴浘锛圧EADME 椤堕儴灞曠ず鐢級
docs/images/                  鏍蜂緥鍥撅紙鐓х墖妯″紡 / 浜屽€兼ā寮忓姣斻€佹暟鎹泦棰勮琛級
tools/gen_default_config.py   閲嶆柊鐢熸垚榛樿閰嶇疆锛坄--check` 鏍￠獙鏄惁婕傜Щ锛?tools/make_preview_sheet.py   鎶婃暟鎹泦娓叉煋鎴愬姣斿浘锛堜富鎸囩汗 脳 鎹哄嵃 脳 TTA + 鏍囨敞鍙犲姞锛?tools/shoot_gui.py            绂诲睆娓叉煋 8 涓〉闈负 PNG锛堟棤鏄剧ず鍣ㄤ篃鑳芥鏌ョ晫闈級
tools/launch_gui.vbs          鏃犻粦鑹叉帶鍒跺彴绐楀彛鍚姩鐣岄潰锛坰tart_gui.bat 璋冪敤鍚屼竴璺緞锛?tools/cleanup_locked.ps1      绠＄悊鍛樻竻鐞嗗彈闄愭矙绠遍仐鐣欑殑涓嶅彲鍒犻櫎绌虹洰褰?```

渚濊禆鏂瑰悜鏄?*鍗曞悜**鐨勶細`gui 鈫?app 鈫?{pipeline, core} 鈫?plugins 鈫?legacy`銆?`core` 涓?`pipeline` 瀹屽叏涓嶄緷璧?Qt锛屽洜姝ゅ彲浠ュ湪鏃犲浘褰㈢幆澧冿紙鏈嶅姟鍣?/ CI锛夐噷璺戦€氥€?
---

## 8. 渚濊禆锛堝埢鎰忎繚鎸佹渶灏忥級

| 鍖?| 鐢ㄩ€?| 蹇呰鎬?|
|----|------|--------|
| `numpy` | 鏁扮粍涓庢暟鍊艰绠?| 蹇呴渶 |
| `opencv-python` | 婊ゆ尝 / 閲嶆槧灏?/ 缂栬В鐮?| 蹇呴渶 |
| `pydantic` | 閰嶇疆鏍￠獙锛堝崟涓€浜嬪疄鏉ユ簮锛?| 蹇呴渶 |
| `PyYAML` | 閰嶇疆璇诲啓 | 蹇呴渶 |
| `PySide6` **鎴?* `PyQt6` | 鍥惧舰鐣岄潰锛堜簩鑰呬换閫夊叾涓€鍗冲彲锛?| 蹇呴渶锛圙UI 鏄竴绾у叆鍙ｏ級 |
| `pytest` / `ruff` | 娴嬭瘯涓庨潤鎬佹鏌?| 寮€鍙?|

**Qt 鍙岀粦瀹氭敮鎸?*锛欸UI 閫氳繃 `synthfp/gui/qt.py` 杩欎竴涓吋瀹瑰眰璁块棶 Qt锛?瀹冧紭鍏堢敤 PySide6锛堣璁℃枃妗?3.5 鐨勯€夊瀷锛夛紝鎵句笉鍒板氨鑷姩鍥為€€鍒?PyQt6銆?涓よ竟鏋氫妇鐢ㄦ硶涓€鑷达紝鍙湁淇″彿/妲藉懡鍚嶄笉鍚岋紙`Signal`/`Slot`/`Property`
瀵?`pyqtSignal`/`pyqtSlot`/`pyqtProperty`锛夛紝宸紓鍏ㄩ儴鏀舵暃鍦ㄩ偅涓€涓枃浠堕噷銆?鍥犳锛?
```bash
pip install PySide6     # 鎺ㄨ崘
# 鎴?pip install PyQt6       # 鐜閲屽凡鏈?PyQt6 鏃剁洿鎺ュ彲鐢紝鏃犻渶鍐嶈
```

鐣岄潰浠ｇ爜閲屼换浣曞湴鏂归兘涓嶄細鍑虹幇 `from PySide6 ...` 鎴?`from PyQt6 ...`
锛堟湁鍗曞厓娴嬭瘯瀹堢潃杩欐潯绾︽潫锛夈€?
**鍒绘剰涓嶅紩鍏?*锛歴cipy銆乻cikit-image銆乵atplotlib銆乸andas銆乧lick銆乼qdm銆?qdarktheme銆乸yqtgraph 鈥斺€?闇€瑕佺殑鍔熻兘閮界敤 numpy/opencv 鑷繁瀹炵幇
锛堣川閲忚瘎鍒嗐€佸悜閲忓寲楠ㄦ灦鍖栥€佺粺璁″浘琛ㄣ€佽繘搴︽樉绀猴級銆?
---

## 9. 鎬ц兘

鍦?8 鏍告満鍣ㄤ笂 320脳480 鍗曞紶涓绘寚绾圭殑瀹炴祴锛?
| 閰嶇疆 | 鍗曞紶鑰楁椂 | 璇存槑 |
|------|---------|------|
| 榛樿锛堥鐜囨。鏁?5锛岃凯浠ｅ垎杈ㄧ巼 0.7锛?| **绾?0.7 s** | 鎺ㄨ崘 |
| 杩唬鍒嗚鲸鐜?1.0锛堟渶閿愬埄锛?| 绾?1.3 s | 杩芥眰鏈€楂樿剨绾块攼搴︽椂浣跨敤 |
| 鍘熼」鐩湭浼樺寲瀹炵幇 | 绾?4.8 s | 閫愬儚绱?Zhang-Suen + 鏃犳牳缂撳瓨 + 20 棰戠巼妗?|

涓変釜浼樺寲鐐癸紙绠楁硶涓€琛屾湭鏀癸級锛?
1. **Gabor 鏍稿簱妯″潡绾х紦瀛?* 鈥斺€?720 涓牳鍙瀯寤轰竴娆★紝璺ㄦ牱鏈?璺ㄥ疄渚嬪鐢紱
2. **棰戠巼妗ｆ暟 20 鈫?5** 鈥斺€?鎸囩汗灞€閮ㄨ剨绾垮懆鏈熷彉鍖栧钩缂擄紝5 妗ｈ冻澶燂紱妗ｆ暟鐩存帴鍐冲畾姣忔杩唬鐨勫嵎绉鏁帮紱
3. **杩唬鍒嗚鲸鐜囨瘮渚?0.7 + 绉婚櫎鍐呭祵缁嗗寲** 鈥斺€?鍗风Н閲忕害鍑忓崐锛沍hang-Suen 鏀圭敱缁嗚妭鐐规彃浠跺湪
   **鏈€缁堟牱鏈浘**涓婂仛涓€娆★紙鍚戦噺鍖?LUT 瀹炵幇锛?00脳500 绾?20 ms锛夛紝鑰屼笉鏄瘡涓富鎸囩汗鍋氫竴閬?   O(200脳H脳W) 鐨?Python 寰幆銆?
浠诲姟璋冨害鐢ㄧ嚎绋嬫睜锛氶噸娲绘槸 OpenCV 鐨勬护娉?/ 閲嶆槧灏勶紝杩欎簺璋冪敤浼氶噴鏀?GIL銆?锛堣繘绋嬫睜鍦?Windows 涓婇渶瑕佹帓闃?+ pickle 浼犺緭 numpy 鏁扮粍锛屼笖妯″潡绾ф牳缂撳瓨鏃犳硶璺ㄨ繘绋嬪鐢紝瀹炴祴鏇存參銆傦級

---

## 10. 甯歌闂

**Q锛氱敓鎴愮殑鎸囩汗鐪嬭捣鏉ュ儚銆屽渾瑙掓柟鍧椼€嶏紵**
A锛氶偅鏄?SFinGe 5 鍙傛暟妯″瀷鐨勫浐鏈夎疆寤擄紙鐭╁舰涓 + 涓婁笅妞渾甯斤級銆?榛樿浣跨敤 `backend.shape_style = round`锛堣秴妞渾鎸囪吂杞粨锛夋洿鑷劧锛?闇€瑕佷笌鍘熺増閫愬儚绱犲鐓ф椂鏀规垚 `box`銆?
**Q锛歚tta_index = 0` 涓轰粈涔堟病鏈夌畻瀛愶紵**
A锛氳繖鏄埢鎰忚璁?鈥斺€?姣忎釜鎹哄嵃鐨勭 0 涓牱鏈繚鐣?*鏈彉鎹㈢殑鍘熷鎹哄嵃**浣滀负骞插噣瀵圭収锛?鎵€鏈?TTA 浠?`tta_index >= 1` 寮€濮嬨€傝嫢鍙兂瑕佸彉鎹㈢粨鏋滐紝鎶?`tta_per_impression` 璋冨ぇ鍗冲彲銆?
**Q锛氱敓鎴愪腑鏂簡鎬庝箞鍔烇紵**
A锛氱洿鎺ラ噸璺戝悓鏍风殑鍛戒护銆傝皟搴﹀櫒璇诲彇 `logs/done.jsonl` 骞堕€愭牱鏈‘璁ゅ浘鍍忎粛鍦ㄧ鐩樹笂锛?鍙ˉ鐢熸垚缂哄け鐨勬牱鏈紝鏃ュ織閲屼細鏄剧ず銆屾柇鐐圭画浼狅細璺宠繃 N 涓凡瀹屾垚鏍锋湰銆嶃€?
**Q锛氭兂鎹㈣嚜娲芥ā鍨嬶紙Turing 绛夛級鎬庝箞鍔烇紵**
A锛氬疄鐜?`synthfp/plugins/backends/base.py` 鐨?`FingerprintBackend`
锛堝彧闇€ `generate_master(spec) -> MasterArtifact`锛夛紝鐢?`@BackendRegistry.register` 娉ㄥ唽锛?GUI 鐨勩€屽悗绔厤缃€嶉〉浼氳嚜鍔ㄥ嚭鐜拌鍚庣锛岄厤缃噷 `backend.name` 鏀规垚瀹冨嵆鍙€?
**Q锛氱敓鎴愬嚭鏉ョ殑鑴婄嚎瀵嗗害鍜岀湡瀹炴寚绾逛笉涓€鏍凤紵**
A锛氱湡瀹?500 dpi 鎸囩汗鐨勮剨绾垮懆鏈燂紙涓€鑴?+ 涓€璋凤級绾?**8~12 px**锛?.4~0.6 mm锛夛紝
鑴婄嚎绾﹀崰鎬诲搴︿竴鍗娿€傛湰椤圭洰鎸夎繖涓墿鐞嗛噺鏍囧畾锛屽苟鏈夌敤渚嬪畧鐫€
锛坄tests/test_backend_ridge.py` 鐢ㄨ嚜鐩稿叧娴嬪懆鏈燂紝瓒呭嚭 7~14 px 灏卞け璐ワ級銆?璋冨瘑搴﹁鏀广€屽悗绔厤缃?鈫?鑴婄嚎鍚堟垚 鈫?鏈€绐?鏈€瀹借剨瀹姐€嶏細
鏁板€兼槸**涓€涓懆鏈熷崰澶氬皯鍍忕礌**锛岃皟灏忓彉瀵嗐€佽皟澶у彉寰楃█鐤忋€?
**Q锛氭兂鍙繚鐣欓粦鑹茶剨绾匡紙鍋氬垎鍓?/ 缁嗗寲鐨勬爣娉ㄦ垨杈撳叆锛夛紵**
A锛氬嬀閫夈€屽悗绔厤缃?鈫?澶栬 鈫?浠呰剨绾挎樉绀轰负榛戣壊锛堢函浜屽€煎浘锛夈€嶏紝
鎴栭厤缃?`backend.ridge_only_black: true`銆傝緭鍑哄彧鏈?0 涓?255 涓や釜鍙栧€硷細
鑴婄嚎绾粦銆佽胺绾跨函鐧斤紝绾圭悊涓庡簳鍣竴骞跺叧闂紝涔熶笉鍋氳竟缂樼窘鍖栥€?
![鐓х墖妯″紡涓庝簩鍊兼ā寮忓姣擼(docs/images/sample-photo-vs-binary.png)

宸︿笂锛氶粯璁わ紙鐓х墖锛夋ā寮忥紱鍙充笂锛歚ridge_only_black`锛涗笅涓€琛屾槸灞€閮?2脳 鏀惧ぇ銆?涓ょ妯″紡瀵瑰悓涓€绉嶅瓙瀹屽叏纭畾锛屽洜姝ゅ彲鐢ㄤ簬鍚屼竴韬唤鐨勫浠诲姟鏍囨敞銆?
**Q锛氳川閲忓垎鏄?NFIQ2 鍚楋紵**
A锛氫笉鏄晢鐢?NFIQ2锛岃€屾槸鑷爺鐨?**NFIQ2 椋庢牸**璇勫垎锛堜簲椤瑰姞鏉冿細鑴婄嚎娓呮櫚搴?35% / 瀵规瘮搴?20% /
鏂瑰悜涓€鑷存€?25% / 鏈夋晥闈㈢Н 10% / 鑴婄嚎鍛ㄦ湡鍚堢悊鎬?10%锛夛紝绛夌骇 L1鈥揕5 涓?NFIQ2 鍚屽悜銆?濂藉鏄浂澶栭儴渚濊禆銆佽法骞冲彴涓€鑷淬€佸畬鍏ㄧ‘瀹氾紱manifest 瀛楁鍚嶆部鐢?`nfiq2` 浠ヤ究涓嬫父缁熶竴澶勭悊銆?
---

## 11. 娴嬭瘯

```bash
python -m pytest tests -o addopts="" -q     # 鍏ㄩ噺锛堢害 25 绉掞級
python -m pytest tests/test_tta.py -q       # 鍗曚釜妯″潡
```

褰撳墠鐘舵€侊細**912 passed**銆傝鐩栬寖鍥达細

| 鏂囦欢 | 鐢ㄤ緥 | 鍐呭 |
|------|------|------|
| `test_tta.py` | 465 | 21 涓?TTA 绠楀瓙锛氱‘瀹氭€с€佸€煎煙銆佷笉淇敼杈撳叆銆佸弬鏁板鐜?|
| `test_quality.py` | 68 | 鍏」璐ㄩ噺鎸囨爣 + 缁煎悎璇勫垎 + 鍒嗙骇杈圭晫 |
| `test_impressions.py` | 77 | 鍘嬪姏浣嶇Щ鍦恒€佸彉褰€? 绉嶄紶鎰熷櫒銆佸潗鏍囨槧灏?|
| `test_config.py` | 64 | 棰勮銆佹牎楠屾嫆缁濄€乊AML 寰€杩斾笌骞傜瓑銆佺偣鍙疯鐩栵紙鍚厓缁勪笅鏍囷級銆佸悎骞惰涔?|
| `test_core.py` | 56 | 绉嶅瓙鏍戙€佽鍒掔‘瀹氭€с€佸垝鍒嗐€丮anifest銆佸瓨鍌ㄣ€侀噺鍖?|
| `test_minutiae.py` | 55 | 鍚戦噺鍖?Zhang-Suen銆丆N 妫€娴嬨€佷吉鐐硅繃婊ゃ€両SO 妯℃澘 |
| `test_pipeline.py` | 37 | 绠楀瓙閾俱€佺鍒扮钀界洏銆佸彲澶嶇幇鎬с€佸け璐ラ殧绂讳笌閲嶈瘯銆佷簩鍊兼敹鍙?|
| `test_app.py` | 33 | 鏈嶅姟鍥炶皟銆佹柇鐐圭画浼犮€佸鍑哄箓绛夈€佺粺璁¤嚜娲?|
| `test_cli.py` | 28 | 鍏ㄩ儴瀛愬懡浠ゃ€侀€€鍑虹爜銆侀敊璇矾寰勩€丟BK 鎺у埗鍙拌緭鍑?|
| `test_gui.py` | 22 | 8 椤甸潰鏋勯€犱笌鍒锋柊銆佷富绐楀彛瀵艰埅銆丵t 鍏煎灞傜害鏉熴€丟UI 椹卞姩鏁撮摼璺敓鎴愩€佸瑙傛帶浠跺洖濉?|
| `test_backend_ridge.py` | 7 | 鑴婄嚎鍛ㄦ湡锛堣嚜鐩稿叧 7~14 px锛夈€佸崰绌烘瘮銆佺伆搴﹀姣斻€佷簩鍊兼ā寮?|

### 鍙楅檺娌欑涓嬬殑涓存椂鐩綍

`tests/conftest.py` 浼氬厛鎺㈡祴銆岃兘鍚﹁鍙栦竴涓?0o700 鐩綍銆嶃€傚鏋滀笉琛?锛圵indows 涓婁互鍙楅檺浠ょ墝杩愯鐨勮嚜鍔ㄥ寲娌欑浼氳繖鏍凤級锛屽畠鍙仛涓や欢鏀跺熬宸ヤ綔锛?鎶婂缓鐩綍鐨勬潈闄愪綅鏀惧鍒?0o777锛屽苟鎶?pytest 鐨勪复鏃舵牴鎸埌宸ヤ綔鍖哄唴鐨?`.pytest-work/`銆備袱鑰呴兘鍙綔鐢ㄤ簬**鏈娴嬭瘯杩涚▼**锛屼笉鏀规祴璇曡涔夛紱
鏅€氱幆澧冧笅鎺㈡祴閫氳繃锛屽畬鍏ㄤ笉鍋氭浛鎹€?
鑻?`tests/` 涓嬪嚭鐜?`_sfp_probe_*`銆乣_pytmp`銆乣pytest-cache-files-*` 杩欑被
**杩炶嚜宸遍兘璇讳笉浜?*鐨勭┖鐩綍锛堝彈闄愭矙绠辩殑閬楃暀鐗╋級锛屾櫘閫氬垹闄や細鎶ユ嫆缁濊闂€?鐢ㄧ鐞嗗憳 PowerShell 璺戜竴娆?[tools/cleanup_locked.ps1](tools/cleanup_locked.ps1) 鍗冲彲锛?
```powershell
powershell -ExecutionPolicy Bypass -File tools\cleanup_locked.ps1
```

鑴氭湰鍙垹娓呭崟閲岀殑鍚嶅瓧锛堥€愪釜鏍￠獙蹇呴』浣嶄簬宸ヤ綔鍖哄唴锛夛紝涓嶇浠讳綍椤圭洰鏂囦欢銆?杩欎簺鐩綍涔熷凡鍦?`.gitignore` 涓?pytest 鐨?`norecursedirs` 涓帓闄わ紝
鎵€浠?*鍗充娇鏆傛椂娓呬笉鎺夛紝`pytest tests` 涔熺収甯稿伐浣?*銆?
---

## 12. 鍙傝€冩枃鐚?
1. Cappelli, R. et al. "Synthetic Fingerprint-Image Generation." *ICPR*, 2000.
2. Cappelli, R. "SFinGe: an Approach to Synthetic Fingerprint Generation." *BioAuth Workshop*, 2004.
3. Maltoni, D. et al. *Handbook of Fingerprint Recognition.* Springer, 2009.
4. Sherlock, B.G., Monro, D.M. "A Model for Interpreting Fingerprint Topology." *Pattern Recognition*, 1993.
5. Vizcaya, P.R., Gerhardt, L.A. "A Nonlinear Orientation Model for Partial Fingerprints." *Pattern Recognition*, 1996.
6. Perlin, K. "An Image Synthesizer." *SIGGRAPH*, 1985.
7. Zhang, T.Y., Suen, C.Y. "A Fast Parallel Algorithm for Thinning Digital Patterns." *CACM*, 1984.

---

## 13. 鍙備笌璐＄尞 / 寮曠敤 / 璁稿彲

- 鎯虫敼浠ｇ爜锛熷厛鐪?[CONTRIBUTING.md](CONTRIBUTING.md)锛堢幆澧冦€佺害瀹氥€佹墿灞曠偣锛夛紝
  浠ュ強 [docs/CONTRACTS.md](docs/CONTRACTS.md)锛堟彃浠舵帴鍙ｅ绾︼級銆?- 鐢ㄥ畠浜у嚭浜嗚鏂?鏁版嵁闆嗭紵寮曠敤淇℃伅瑙?[CITATION.cff](CITATION.cff)銆?- 鏈」鐩互 **MIT License** 鍙戝竷锛岀増鏉冨綊 **xf785**锛?  鍏ㄦ枃瑙?[LICENSE](LICENSE)銆?
### 鍏充簬鍚堟垚鏁版嵁鐨勪娇鐢ㄨ竟鐣?
MIT 璁稿彲鎺堜簣浣犲**浠ｇ爜**鐨勫箍娉涜嚜鐢憋紱浣嗚娉ㄦ剰锛氱敓鎴愮殑鍥惧儚鏄?*绠楁硶鍚堟垚鐨?*锛?涓嶅惈浠讳綍鐪熷疄浜虹被鎸囩汗銆傝鍕挎妸瀹冧滑鐢ㄤ簬鍐掑厖鐪熷疄韬唤銆佽閬跨敓鐗╄瘑鍒郴缁燂紝
鎴栧湪鏈０鏄庣殑鍦哄悎鍏呭綋鐪熷疄閲囬泦鏁版嵁銆傚鏈娇鐢ㄦ椂锛屽缓璁湪璁烘枃涓樉寮忔敞鏄庢暟鎹敱鏈伐鍏峰悎鎴?锛堟暟鎹泦閲岀殑 `README.md` 涓?`environment.json` 宸茶嚜鍔ㄥ甫涓婅澹版槑涓庡彲澶嶇幇淇℃伅锛夈€?

