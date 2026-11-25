# Minh phụ trách lệnh tái lập và bàn giao. Lan Anh phụ trách ViT/ablation, Hiền phụ trách đánh giá cuối.
param(
    [string]$Python = "python",
    [string]$CleanPython = "runs/week4_env/Scripts/python.exe",
    [string]$Tectonic = "runs/week4_tex/tectonic.exe",
    [switch]$ExecuteTest
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# Minh chạy từng bước bằng cùng Python và dừng nếu một bước trả lỗi.
function Invoke-Module([string]$Interpreter, [string]$Module, [string[]]$Options) {
    & $Interpreter -X utf8 -m $Module @Options
    if ($LASTEXITCODE -ne 0) { throw "Module $Module exited with $LASTEXITCODE" }
}

function Build-Tex([string]$Entry) {
    if (Test-Path -LiteralPath $Tectonic) {
        & $Tectonic -X compile $Entry
        if ($LASTEXITCODE -ne 0) { throw "Tectonic failed for $Entry" }
        return
    }
    Push-Location (Split-Path $Entry)
    try {
        $texFile = Split-Path -Leaf $Entry
        $texName = [IO.Path]::GetFileNameWithoutExtension($texFile)
        & lualatex -halt-on-error -interaction=batchmode $texFile
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed for $Entry" }
        if ($Entry.StartsWith("reports/")) {
            & bibtex $texName
            if ($LASTEXITCODE -ne 0) { throw "BibTeX failed for $Entry" }
        }
        & lualatex -halt-on-error -interaction=batchmode $texFile
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed for $Entry" }
        & lualatex -halt-on-error -interaction=batchmode $texFile
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed for $Entry" }
    } finally {
        Pop-Location
    }
}

# Các trainer dùng last.pt để resume hoặc kiểm tra lượt đã hoàn tất.
foreach ($model in @("vit", "ablation")) {
    Invoke-Module $Python scripts.run_lanh_week3 @("--stage","train","--model",$model,"--device","cuda")
    Invoke-Module $Python scripts.run_lanh_week3 @("--stage","collect","--model",$model,"--device","cuda")
}
foreach ($model in @("resnet", "swin")) {
    Invoke-Module $Python scripts.run_minh_week3 @("--stage","train","--model",$model)
    Invoke-Module $Python scripts.run_minh_week3 @("--stage","collect","--model",$model)
}
Invoke-Module $Python scripts.reproduce_week4 @("--strict")

$repeatDir = "runs/week4_repeat/resnet_10_seed42"
& $CleanPython -X utf8 -c "from pathlib import Path; from scripts.run_minh_week2 import run_one; run_one(Path('configs/week2/resnet_10.yaml'), Path('$repeatDir'), 'cuda', resume=True)"
if ($LASTEXITCODE -ne 0) { throw "Clean-environment training failed" }
$options = @("--out","results/minh_week4","--repeat",$repeatDir)
foreach ($csv in @("results/team_week2/p0_runs.csv","results/week3/resnet_runs.csv",
                    "results/week3/vit_runs.csv","results/week3/ablation_runs.csv","results/week3/swin_runs.csv")) {
    $options += @("--records",$csv)
}
Invoke-Module $CleanPython scripts.handoff_week4 $options
Invoke-Module $CleanPython src.predict @("--image","results/minh_week4/demo_input.png",
    "--checkpoint","runs/week2/resnet_d100_s42/best.pt","--out","results/minh_week4/demo_resnet.json","--device","cpu")
Invoke-Module $Python scripts.analyze_hien_week3 @(
    "--csv","results/minh_week4/baseline_18_val_runs.csv",
    "--out","results/minh_week4/paired",
    "--run","runs/week2/resnet_d50_s42","--run","runs/week2_vit_50_seed42",
    "--run","runs/week3/resnet_50_seed43","--run","runs/week3/vit_50_seed43",
    "--run","runs/week3/resnet_50_seed44","--run","runs/week3/vit_50_seed44")
Invoke-Module $Python scripts.evaluate_final
if ($ExecuteTest) {
    if (Test-Path -LiteralPath "results/final_test") {
        # Kết quả chỉ được dùng lại khi khớp danh sách đã khóa và bảng đếm đủ 18 lượt.
        Invoke-Module $Python scripts.week4_tables @("--check-test")
    } else {
        Invoke-Module $Python scripts.evaluate_final @("--execute","--device","cuda")
    }
    Invoke-Module $Python scripts.reproduce_week4 @("--strict")
    Invoke-Module $Python scripts.week4_tables
    # Xuất đúng bản PDF dùng các bảng vừa sinh, rồi mới đóng ZIP báo cáo của Hiền.
    foreach ($entry in @("reports/final.tex", "reports/milestone.tex", "slides/week4.tex")) {
        Build-Tex $entry
    }
    Invoke-Module $Python scripts.package_hien_week4
    Invoke-Module $CleanPython pytest @("-q")
}
