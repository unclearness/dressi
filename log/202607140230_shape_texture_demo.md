# shape_texture_optimization: UV球 → バニー（形状→テクスチャの2段デモ）

## Goal
UV展開された球をバニーにフィットさせる新デモ。まずシルエット最適化で形状を
合わせ、続けてそのフィット済み球にテクスチャアトラスを最適化する2フェーズ構成。

## What was done
- 新example `examples/shape_texture_optimization/`（run.h / main.cpp / run.cpp）を追加。
  registry / examples/CMakeLists.txt に配線（geometryShader 必須なので
  `needs_geometry_shader = true`）。
- 共通ジオメトリに3関数を追加（`examples/common/geometry_utils.{h,cpp}`）:
  - `GenerateUvSphere(stacks, slices)`: 緯度経度UV球（アンウェルド、シーム・極を
    複製、equirectangular UV = (phi/2pi, theta/pi)、極の縮退三角形は省略）。
  - `WeldByPosition(mesh)`: 位置量子化(1e-5)で溶接。`pos`/`faces`/`to_welded`
    （元頂点→溶接頂点の写像）を返し、縮退面を除去。
  - `SaveTexturedObjMesh`: v/vt + f v/vt でOBJ出力。
- Phase 1（形状）: silhouette_optimization の **AA技法**パスを移植。多視点AA
  シルエット + 一様Laplacian(0.5) + normal-consistency(0.5) + in-graph Adam。
  最適化対象は**溶接済み**球（UV不要）。
- Phase 2（テクスチャ）: texture_optimization を2メッシュ化。溶接済みの最適化
  位置を `to_welded` でアンウェルドUV球に散布して形状を固定し、球のUV空間の
  アトラス（512x256）を最適化。ターゲットは同一Sobolジッタ下でin-graph描画した
  テクスチャ付きバニー。損失はバニーmaskでゲート（球のはみ出しフリンジが黒を
  学習しないように）。

## Key decision: なぜ溶接（2表現）が必要か
「OBJスタイルの1頂点複数UVで十分では？」への回答 = **それが実体だが、dressiの
GPUリーフ制約で2表現に展開が必須**:
1. UV格納側: dressiのUVは頂点属性 `{V,1,2}` で位置と同じface indexで参照する
   ため「1頂点=1UV」。OBJの `f v/vt` は GPU に載せる時点で (pos,uv) 一意ペアごとに
   頂点を割る（=アンウェルド、LoadObjMesh と同じ）必要 → テクスチャ相はアンウェルド。
2. 形状勾配側: シーム頂点は隣接面がシーム左右に割れ、複製A/Bはそれぞれ半分の
   星からしか勾配を受けない → 別々に動きシームが裂ける。位置を共有した1頂点
   （=OBJの `v`）で最適化する必要 → 形状相は溶接。
   → `sphere`(展開=v/vt) + `sphere_w`(溶接=v) + `to_welded`(v/vt→v) は
   OBJ表現そのもの。追加コストは小さな index マップのみ。

## Measurements（スモーク: shape 40 / texture 120 iter, 8 views）
- UV球 2145v/3968f（溶接 1986v/3968f）、バニー 16323v/30338f、atlas 1024².
- Phase1: loss 2.09 → 0.25、mean silhouette IoU 0.887（40iterのため耳未成長。
  既定200iterで改善見込み）。
- Phase2: loss 0.97 → 0.078。七宝焼き調の花柄・色が球に転写されるのを目視確認。

## Fixes (フォローアップ)
- テクスチャ既定 iter を 1200 → 300 に短縮。
- `GenerateUvSphere` の面の巻きが内向き（法線が内側）だった → 頂点順を反転して
  外向き(CCW from outside)に。ラスタライザはカリング無しなのでシルエット相/描画
  は不変、出力OBJの法線のみ正される。
- `SaveTexturedObjMesh` に mtl 出力を追加（`mtllib`/`usemtl` + `map_Kd`）。OBJの
  V座標は `1-v` で反転（dressi/Vulkanは左上原点=行0、OBJ/GLサンプラはv=0が下）。
  → MeshLab でテクスチャ付きで開ける。引数に texture_filename を追加。

## Redesign: 形状はアイコスフィアで最適化 → UV球へ方向転写
UV球を直接シルエット最適化すると、極の引き伸ばされた三角形・不均一な頂点密度の
ため形状品質が silhouette-only(アイコスフィア)より劣化した（IoU 0.924）。
- 形状フェーズを**アイコスフィア**（level 4、均一な変形ケージ）に変更。
- `WeldByPosition` を廃止し `TransferDeformationByDirection` を追加:
  各UV球頂点の方向で原点からのレイをアイコスフィアのrest三角形に当て
  (Moeller-Trumbore)、そのバリセントリック重みを**変形後**アイコスフィア頂点に
  適用してUV球へ変形を転写。方向の関数なのでシーム/極の複製頂点（同方向）は
  同一位置に写り、**溶接なしでシームが閉じる**（前回の「OBJスタイルだけで十分」
  という指摘をこの設計が満たす）。
- 結果: mean silhouette IoU 0.924 → **0.9635**（silhouette-only ~0.951-0.956 と同等）。
  耳まで綺麗に成形、テクスチャ相の花柄定位も鮮明に。
- 新オプション `--sphere-level`（アイコスフィア分割、既定4）。

## Android 対応
- レジストリ登録済みなので `NativeBridge.listExamples()` 経由でボタンは自動表示。
  needs_geometry_shader=true でAAが使う gl_PrimitiveID 非対応端末では自動グレーアウト。
- `MainActivity.kt`: `defaultArgs` に `shape_texture_optimization`→
  `--data-dir=<bunny> --out-dir=<out>` を追加。`missingAsset` の bunny 分岐に追加し
  bunny.obj + bunny-atlas.jpg の存在を確認（両方 syncDressiAssets で同梱済み）。
- **ビューアをフェーズ間で共有する設計に変更**（Android対応の要）: AndroidHost の
  `registerStream` は破棄時に解除しないため、フェーズ毎にビューアを作ると stream
  ボタンが累積し、phase2開始時に選択streamが死んで画面が固まる。→ `Viewers` 構造体
  を導入し、Run本体で target/pred/tex の3ビューアを1度だけ生成して両フェーズに
  ポインタ渡し。全タイル出力が2:1なので単一ウィンドウのアスペクトで両フェーズを
  カバー（デスクトップビューアは任意サイズ画像をスケール表示）。
- ビルド確認: `gradlew :app:assembleDebug` 成功（arm64-v8a、新run.cppがcore lib経由で
  libdressi_examples.so に取り込まれる）。デスクトップも refactor 後に再検証（IoU 0.929
  @60iter）。**実機での実行確認は端末未接続のため未実施**。

## Android: 最適化終了後もstream切替を可能に（全例共通）
問題: 最適化ループ終了後は `update()` が呼ばれず、streamボタンを押しても再描画
されない（ボタンが効かない）。原因は AndroidHost が「選択stream以外は描画しない・
フレームを保持しない」設計だったため。
- `SurfaceState` に **per-stream フレームキャッシュ** `m_frames`（m_titles と並行）を追加。
- `blit()`: 毎回そのstreamの最新フレームをキャッシュし、選択中のときだけ描画。描画
  本体は `drawLocked()`（mutex保持前提のヘルパー）に切り出して再利用。
- `selectStream()`: 選択変更時にキャッシュ済みフレームを**即再描画** → 終了後でも
  ボタンでtarget/最適化先を切り替えられる。
- `setSurface()`: サーフェス再生成時も選択streamのキャッシュを再描画（黒画面回避）。
- `registerStream`/`clearStreams` を m_frames と整合（実行開始時 jni_bridge が
  clearStreams を呼ぶので前回フレームは残らない）。
- 共有ホストの修正なので image_fitting/texture/silhouette/pbr 全例に適用。Debug/
  Release 両APKビルド成功。実機確認は端末未接続のため未実施。

## Android: デフォルト選択を最適化結果に + 選択ボタンの色付け（全例共通）
- 全例が `makeViewer` を target(0) → 最適化結果pred(1) → 補助(tex/env, 2) の順で
  生成しており、index 1 が常に「最適化対象」。デフォルト選択を index 1（無ければ0）に:
  - ネイティブ `registerStream`: 2番目のstream登録時 `m_selected = 1`（早期フレームの
    ちらつき防止、warmup前からpredを表示）。
  - Kotlin `onStreamsChanged`: `selectedStream = if(size>1) 1 else 0`、`selectStream(既定)`
    でネイティブと同期。
- 選択中streamボタンを青背景+白文字でハイライト（`styleStreamButton`）、タップで
  `selectStreamButton` が再スタイル。`streamButtons` リストで管理。
- Debug/Release 両APKビルド成功。実機確認は端末未接続のため未実施。

## Open issues
- 既定（shape 200 / texture 1200）でのフル品質・定常ms・IoUは未計測。README/
  docs/benchmarks.md への追記は未（bench.json は本デモ未出力＝計測メソッド未統合）。
- Android UI一覧への追加は未確認（core libはビルド可、registryには登録済み）。
- シーム/極の複製に起因するアトラスのシーム近傍テクセルは、UV球の性質上やや
  低密度（テクスチャ相のジッタ被覆に依存）。
